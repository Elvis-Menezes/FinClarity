#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------
# EnsembleClusterRetriever (with retrieval evaluation)
# ---------------------------------------------------------------------
class EnsembleClusterRetriever:
    """
    A document retriever built on top of TF-IDF vectors and a clustering model.
    After fitting on a corpus, it answers queries by first routing to a few
    likely clusters (by centroid similarity) and then ranking docs inside
    those clusters by cosine (dot) similarity.
    """

    def __init__(self, n_clusters: int = 30, top_clusters: int = 1,
                 temperature: float = 0.25, minibatch: bool = True,
                 random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.top_clusters = top_clusters
        self.temperature = temperature
        self.random_state = random_state
        self.minibatch = minibatch

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.7, min_df=2)
        self.km = None
        self.X = None              # (n_docs, d) normalized TF-IDF
        self.labels_ = None        # cluster assignment per doc
        self.centroids_ = None     # (k, d) normalized centroids
        self.docs: Optional[List[str]] = None
        self.meta: Optional[List] = None

    @staticmethod
    def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
        z = (x - x.max()) / max(1e-9, temp)
        ez = np.exp(z)
        return ez / (ez.sum() + 1e-9)

    def fit(self, docs: List[str], meta: Optional[List] = None) -> None:
        """
        Vectorize and cluster documents. L2-normalize vectors and centroids.
        """
        self.docs = docs
        self.meta = meta

        X = self.vectorizer.fit_transform(docs)
        X = normalize(X, norm="l2", copy=False)
        self.X = X

        KM = MiniBatchKMeans if self.minibatch else KMeans
        # scikit-learn <=1.3 requires an integer for n_init
        self.km = KM(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.labels_ = self.km.fit_predict(X)

        C = self.km.cluster_centers_
        C = normalize(C, norm="l2", copy=False)
        self.centroids_ = C

    def route_clusters(self, q_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a set of top cluster ids and a softmax weight per cluster.
        """
        cent_sim = (self.centroids_ @ q_vec.T).ravel()
        top_idx = np.argsort(-cent_sim)[:self.top_clusters]
        weights = self._softmax(cent_sim[top_idx], temp=self.temperature)
        return top_idx, weights

    def query(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve top_k documents for a free-text query.
        """
        q = self.vectorizer.transform([query])
        q = normalize(q, norm="l2", copy=False)

        top_clusters, weights = self.route_clusters(q)
        scores = []
        for cid, w in zip(top_clusters, weights):
            mask = (self.labels_ == cid)
            if not np.any(mask):
                continue
            Xc = self.X[mask]
            idxs = np.where(mask)[0]
            doc_sim = (Xc @ q.T).toarray().ravel() 
            final = w * doc_sim
            for i, s in zip(idxs, final):
                scores.append((i, float(s), int(cid)))

        scores.sort(key=lambda x: -x[1])
        picks = scores[:top_k]
        out = []
        for r, (i, s, c) in enumerate(picks):
            out.append({
                "rank": r + 1,
                "doc_index": i,
                "score": s,
                "cluster": c,
                "text": None if self.docs is None else self.docs[i],
                "meta": None if (self.meta is None or i >= len(self.meta)) else self.meta[i],
            })
        return out

    def best_cluster_for(self, query: str) -> Tuple[int, float]:
        """
        Return (best_cluster_id, similarity) for the query.
        """
        q = self.vectorizer.transform([query])
        q = normalize(q, norm="l2", copy=False)
        cent_sim = (self.centroids_ @ q.T).ravel()
        best = int(np.argmax(cent_sim))
        return best, float(cent_sim[best])

    # -------------------- Evaluation utilities --------------------
    @staticmethod
    def _precision_at_k(rel_idx: List[int], ranked_idx: List[int], k: int) -> float:
        if k <= 0: return 0.0
        topk = ranked_idx[:k]
        return len(set(topk).intersection(rel_idx)) / float(k)

    @staticmethod
    def _average_precision_at_k(rel_idx: List[int], ranked_idx: List[int], k: int) -> float:
        hits = 0
        score = 0.0
        for i, idx in enumerate(ranked_idx[:k], start=1):
            if idx in rel_idx:
                hits += 1
                score += hits / i
        denom = max(1, len(set(rel_idx)))
        return score / denom

    @staticmethod
    def _ndcg_at_k(rel_idx: List[int], ranked_idx: List[int], k: int) -> float:
        rel_set = set(rel_idx)
        gains = [1.0 if idx in rel_set else 0.0 for idx in ranked_idx[:k]]
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted(gains, reverse=True)
        idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
        return (dcg / idcg) if idcg > 0 else 0.0

    def evaluate_retrieval(self, labels: List[str], top_k: int = 10,
                           metrics_k: int = 10, min_docs: int = 5) -> Dict[str, float]:
        """
        Evaluate macro P@5, MAP@metrics_k and nDCG@metrics_k over label groups.
        Each unique label is treated as a query; docs sharing that label are relevant.
        """
        if self.docs is None or self.X is None:
            raise RuntimeError("Model must be fitted before evaluation.")
        if len(labels) != len(self.docs):
            raise ValueError("Length of labels must match number of documents.")

        unique_labels = sorted(set(labels))
        scores = {"P@5": [], f"MAP@{metrics_k}": [], f"nDCG@{metrics_k}": []}

        for lab in unique_labels:
            rel_idx = [i for i, lb in enumerate(labels) if lb == lab]
            if len(rel_idx) < min_docs:
                continue
            ranked_idx = [r["doc_index"] for r in self.query(lab, top_k=top_k)]
            scores["P@5"].append(self._precision_at_k(rel_idx, ranked_idx, 5))
            scores[f"MAP@{metrics_k}"].append(self._average_precision_at_k(rel_idx, ranked_idx, metrics_k))
            scores[f"nDCG@{metrics_k}"].append(self._ndcg_at_k(rel_idx, ranked_idx, metrics_k))

        macro = {k: float(np.mean(v)) if v else float("nan") for k, v in scores.items()}
        return macro

# ---------------------------------------------------------------------
# Weak company labelling (same idea as your evaluation script)
# ---------------------------------------------------------------------
COMPANY_KEYWORDS: Dict[str, List[str]] = {
    "Reliance Industries": ["reliance", "ril", "jio"],
    "HDFC Bank": ["hdfc bank", "hdfc"],
    "ICICI Bank": ["icici bank", "icici"],
    "Infosys": ["infosys"],
    "Tata Consultancy Services": ["tcs", "tata consultancy"],
    "Hindustan Unilever": ["hindustan unilever", "hul"],
    "Bharti Airtel": ["bharti airtel", "airtel"],
    "State Bank of India": ["state bank of india", "sbi"],
    "Larsen & Toubro": ["larsen", "toubro", "larsen toubro", "lt"],
    "Bajaj Finance": ["bajaj finance"],
    "Axis Bank": ["axis bank", "axis"],
    "Kotak Mahindra Bank": ["kotak", "kotak mahindra"],
    "Maruti Suzuki India": ["maruti", "maruti suzuki"],
    "Asian Paints": ["asian paints"],
    "ITC": ["itc"],
    "NTPC": ["ntpc"],
    "Mahindra & Mahindra": ["mahindra", "mahindra and mahindra", "m&m", "mm"],
    "Sun Pharmaceutical Industries": ["sun pharma", "sun pharmaceutical"],
    "Tata Steel": ["tata steel"],
    "Power Grid Corporation of India": ["power grid"],
    "HCL Technologies": ["hcl", "hcl tech", "hcl technologies"],
    "Titan Company": ["titan"],
    "UltraTech Cement": ["ultratech", "ultra tech"],
    "Wipro": ["wipro"],
    "Nestle India": ["nestle", "nestle india"],
    "Tata Motors": ["tata motors"],
    "JSW Steel": ["jsw steel", "jsw"],
    "Tech Mahindra": ["tech mahindra"],
    "Bajaj Finserv": ["bajaj finserv"],
    "IndusInd Bank": ["indusind", "indusind bank"],
}

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def weak_label_company(headline: str) -> str:
    ct = _norm(headline or "")
    best, best_len = None, -1
    for comp, kws in COMPANY_KEYWORDS.items():
        for kw in kws:
            k = _norm(kw)
            if k and k in ct and len(k) > best_len:
                best, best_len = comp, len(k)
    return best or "Unknown"

# ---------------------------------------------------------------------
# CLI / Demo
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="sensex_finance_news_daily.csv",
                    help="CSV with at least a 'headline' column.")
    ap.add_argument("--clusters", type=int, default=30, help="number of k-means clusters")
    ap.add_argument("--topk", type=int, default=10, help="top K docs to retrieve per query")
    args = ap.parse_args()

    # 1) Load data
    df = pd.read_csv(args.csv)
    if "headline" not in df.columns:
        raise ValueError(f"CSV must contain 'headline'. Found: {list(df.columns)}")
    df = df.dropna(subset=["headline"]).drop_duplicates(subset=["headline"]).reset_index(drop=True)

    # 2) Weak labels (skip 'Unknown')
    labels_all = df["headline"].astype(str).apply(weak_label_company).tolist()
    mask = [lab != "Unknown" for lab in labels_all]
    docs = df.loc[mask, "headline"].astype(str).tolist()
    labels = [lab for lab, m in zip(labels_all, mask) if m]

    if len(docs) < 50:
        print(f"[WARN] Only {len(docs)} labelled rows; results may be noisy.")

    # 3) Fit retriever
    k = min(args.clusters, max(2, len(docs)))  # safe bounds
    retriever = EnsembleClusterRetriever(n_clusters=k, top_clusters=3, temperature=0.25,
                                         minibatch=True, random_state=42)
    retriever.fit(docs, meta=labels)

    # 4) Evaluate retrieval
    metrics = retriever.evaluate_retrieval(labels, top_k=args.topk, metrics_k=10, min_docs=5)
    print("\n=== Macro retrieval metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 5) Quick interactive example
    example_query = "NTPC"
    results = retriever.query(example_query, top_k=50)
    print(f"\nTop 5 for query: {example_query}")
    for r in results:
        print(f"[{r['rank']}] ({r['score']:.4f}) {r['text']}  |  label={r['meta']} ")

if __name__ == "__main__":
    main()
