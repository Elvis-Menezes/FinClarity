#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlined Vectorizer Evaluation for FinClarity Headlines
======================================================
Compares TF-IDF, Word2Vec-CBOW, and Word2Vec-Skip-gram on finance headlines.

What it does
------------
1) Loads a CSV with a "headline" column (default: sensex_finance_news_daily.csv)
2) Cleans & tokenizes text
3) Builds embeddings:
   - TF-IDF (sparse -> dense via toarray)
   - Word2Vec (CBOW & Skip-gram) with mean pooling
4) Creates weak company labels from keyword rules
5) Evaluates retrieval (macro P@5, MAP@10, nDCG@10)
6) (Optional) Evaluates downstream classification (5-fold macro F1)
7) Saves metrics CSV + bar charts + PCA scatter plots

Usage
-----
pip install numpy pandas matplotlib scikit-learn gensim nltk

python streamlined_vector_eval.py \
  --csv sensex_finance_news_daily.csv \
  --save_dir out_eval \
  --max_features 5000 --w2v_dim 100 --w2v_epochs 8 --topk 10 \
  --do_clf

Outputs (in save_dir)
---------------------
- vectorizer_metrics.csv            (macro metrics per model)
- bar_P@5.png, bar_MAP@10.png, bar_nDCG@10.png
- pca_tfidf.png, pca_cbow.png, pca_skip.png
- clf_f1.txt (if --do_clf enabled)
"""

from __future__ import annotations

import argparse
import os
import re
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec


# ----------------------------
# Config & CLI
# ----------------------------
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

@dataclass
class EvalConfig:
    csv_path: str
    save_dir: str
    max_features: int = 5000
    w2v_dim: int = 100
    w2v_window: int = 5
    w2v_epochs: int = 8
    w2v_min_count: int = 1
    topk: int = 10
    do_clf: bool = False
    seed: int = 42

def parse_args() -> EvalConfig:
    ap = argparse.ArgumentParser(description="Streamlined evaluation for TF-IDF vs Word2Vec (CBOW/Skip)")
    ap.add_argument("--csv", dest="csv_path", type=str, default="sensex_finance_news_daily.csv",
                    help="Input CSV with a 'headline' column.")
    ap.add_argument("--save_dir", type=str, default="out_eval", help="Directory to save outputs")
    ap.add_argument("--max_features", type=int, default=5000)
    ap.add_argument("--w2v_dim", type=int, default=100)
    ap.add_argument("--w2v_window", type=int, default=5)
    ap.add_argument("--w2v_epochs", type=int, default=8)
    ap.add_argument("--w2v_min_count", type=int, default=1)
    ap.add_argument("--topk", type=int, default=10, help="k for MAP@k / nDCG@k")
    ap.add_argument("--do_clf", action="store_true", help="Run 5-fold logistic regression (macro F1)")
    args = ap.parse_args()
    return EvalConfig(**vars(args))


# ----------------------------
# Utils
# ----------------------------
def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def try_download_nltk() -> None:
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
        nltk.download("punkt")

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def preprocess_tokens(text: str, stop_words: set) -> List[str]:
    text = normalize_text(text or "")
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in stop_words and len(t) > 2]


# ----------------------------
# Data & labeling
# ----------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "headline" not in df.columns:
        raise ValueError(f"CSV must have a 'headline' column; found {list(df.columns)}")
    df = df.dropna(subset=["headline"]).drop_duplicates(subset=["headline"]).reset_index(drop=True)
    return df

def weak_label_company(headline: str) -> str:
    ct = normalize_text(headline)
    best, best_len = None, -1
    for comp, kws in COMPANY_KEYWORDS.items():
        for kw in kws:
            k = normalize_text(kw)
            if k and k in ct and len(k) > best_len:
                best, best_len = comp, len(k)
    return best or "Unknown"


# ----------------------------
# Vectorization
# ----------------------------
def build_tfidf_embeddings(texts: List[str], max_features: int) -> Tuple[np.ndarray, TfidfVectorizer]:
    vec = TfidfVectorizer(max_features=max_features)
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    return X, vec

def train_w2v(sentences: List[List[str]], dim: int, window: int, epochs: int, min_count: int, sg: int) -> Word2Vec:
    # Word2Vec uses float32 internally
    return Word2Vec(sentences=sentences, vector_size=dim, window=window,
                    min_count=min_count, workers=4, sg=sg, epochs=epochs)

def mean_pool(tokens: List[str], model: Word2Vec) -> np.ndarray:
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)

def embed_sentences(sentences: List[List[str]], model: Word2Vec) -> np.ndarray:
    return np.vstack([mean_pool(toks, model) for toks in sentences]).astype(np.float32)


# ----------------------------
# Evaluation
# ----------------------------
def precision_at_k(rel_idx, ranked_idx, k: int) -> float:
    if k == 0: return 0.0
    topk = ranked_idx[:k]
    return len(set(topk) & set(rel_idx)) / k

def average_precision_at_k(rel_idx, ranked_idx, k: int) -> float:
    hits, score = 0, 0.0
    for i, idx in enumerate(ranked_idx[:k], start=1):
        if idx in rel_idx:
            hits += 1
            score += hits / i
    denom = max(1, len(set(rel_idx)))
    return score / denom

def ndcg_at_k(rel_idx, ranked_idx, k: int) -> float:
    rel_set = set(rel_idx)
    gains = [1.0 if idx in rel_set else 0.0 for idx in ranked_idx[:k]]
    dcg = sum(g / math.log2(i+2) for i, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i+2) for i, g in enumerate(ideal))
    return (dcg / idcg) if idcg > 0 else 0.0

def ranked_indices(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    sims = cosine_similarity(query_vec.reshape(1, -1), matrix)[0]
    return np.argsort(-sims)

def query_vector(model_name: str, company: str,
                 tfidf_vec: TfidfVectorizer,
                 cbow_model: Word2Vec,
                 skip_model: Word2Vec,
                 tokenizer,
                 ) -> np.ndarray:
    toks = tokenizer(company)
    if model_name == "TFIDF":
        return tfidf_vec.transform([" ".join(toks)]).toarray()[0].astype(np.float32)
    elif model_name == "CBOW":
        return mean_pool(toks, cbow_model)
    elif model_name == "SKIP":
        return mean_pool(toks, skip_model)
    else:
        raise ValueError("model_name must be TFIDF/CBOW/SKIP")

def evaluate_retrieval(labels: np.ndarray,
                       tfidf_mat: np.ndarray, tfidf_vec: TfidfVectorizer,
                       cbow_mat: np.ndarray,  cbow_model: Word2Vec,
                       skip_mat: np.ndarray,  skip_model: Word2Vec,
                       tokenizer,
                       topk: int) -> pd.DataFrame:
    models = {
        "TFIDF": (tfidf_mat, tfidf_vec),
        "CBOW":  (cbow_mat,  cbow_model),
        "SKIP":  (skip_mat,  skip_model),
    }
    rows = []
    companies = sorted(np.unique(labels))
    for comp in companies:
        rel_idx = np.where(labels == comp)[0]
        if len(rel_idx) < 5:
            continue  # skip tiny groups
        for name, (mat, helper) in models.items():
            qv = query_vector(name, comp, tfidf_vec, cbow_model, skip_model, tokenizer)
            ranked = ranked_indices(qv, mat)
            rows.append({
                "company": comp,
                "model": name,
                "P@5":     precision_at_k(rel_idx, ranked, 5),
                "MAP@10":  average_precision_at_k(rel_idx, ranked, topk),
                "nDCG@10": ndcg_at_k(rel_idx, ranked, topk),
            })
    return pd.DataFrame(rows)


# ----------------------------
# Plotting
# ----------------------------
def bar_plot(df_summary: pd.DataFrame, metric: str, out_path: str) -> None:
    plt.figure()
    x = df_summary["model"].tolist()
    y = df_summary[metric].tolist()
    plt.bar(x, y)
    plt.title(f"Model comparison — {metric}")
    plt.xlabel("Model"); plt.ylabel(metric)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def pca_scatter(X: np.ndarray, y: np.ndarray, title: str, out_path: str, max_points: int = 2000) -> None:
    n = X.shape[0]
    idx = np.arange(n)
    if n > max_points:
        idx = np.random.choice(idx, size=max_points, replace=False)
    Xs, ys = X[idx], y[idx]
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)
    plt.figure()
    # show up to 12 companies for readability
    uniq = pd.Series(ys).unique().tolist()
    random.shuffle(uniq)
    for lab in uniq[:12]:
        m = (ys == lab)
        plt.scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.7, label=lab)
    plt.title(title); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(markerscale=1, fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()


# ----------------------------
# Main
# ----------------------------
def main(cfg: EvalConfig) -> None:
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    ensure_dirs(cfg.save_dir)
    try_download_nltk()
    sw = set(stopwords.words("english"))

    # 1) Load & preprocess
    df = load_data(cfg.csv_path)
    df["tokens"] = df["headline"].astype(str).apply(lambda s: preprocess_tokens(s, sw))
    df = df[df["tokens"].apply(len) > 0].reset_index(drop=True)
    texts = [" ".join(t) for t in df["tokens"].tolist()]

    # 2) Weak labels
    df["company"] = df["headline"].astype(str).apply(weak_label_company)
    labeled = df[df["company"] != "Unknown"].reset_index(drop=True)
    if len(labeled) < 100:
        print(f"[WARN] Only {len(labeled)} weak-labeled rows — results may be noisy.")

    # 3) Vectorize
    tfidf_mat, tfidf_vec = build_tfidf_embeddings(texts, cfg.max_features)
    cbow_model = train_w2v(df["tokens"].tolist(), cfg.w2v_dim, cfg.w2v_window, cfg.w2v_epochs, cfg.w2v_min_count, sg=0)
    skip_model = train_w2v(df["tokens"].tolist(), cfg.w2v_dim, cfg.w2v_window, cfg.w2v_epochs, cfg.w2v_min_count, sg=1)
    cbow_mat = embed_sentences(df["tokens"].tolist(), cbow_model)
    skip_mat = embed_sentences(df["tokens"].tolist(), skip_model)

    # Align to labeled subset
    mask = df.index.isin(labeled.index)
    tfidf_l = tfidf_mat[mask]
    cbow_l  = cbow_mat[mask]
    skip_l  = skip_mat[mask]
    labels  = labeled["company"].values.astype(str)

    # 4) Retrieval evaluation
    metrics = evaluate_retrieval(labels, tfidf_l, tfidf_vec, cbow_l, cbow_model, skip_l, skip_model,
                                 tokenizer=lambda s: preprocess_tokens(s, sw), topk=cfg.topk)
    macro = metrics.groupby("model", as_index=False)[["P@5", "MAP@10", "nDCG@10"]].mean().round(4)
    macro_path = os.path.join(cfg.save_dir, "vectorizer_metrics.csv")
    macro.to_csv(macro_path, index=False)
    print("\n=== Retrieval (macro over companies) ===")
    print(macro)
    print(f"[OK] Saved: {macro_path}")

    # 5) Plots
    bar_plot(macro, "P@5",     os.path.join(cfg.save_dir, "bar_P@5.png"))
    bar_plot(macro, "MAP@10",  os.path.join(cfg.save_dir, "bar_MAP@10.png"))
    bar_plot(macro, "nDCG@10", os.path.join(cfg.save_dir, "bar_nDCG@10.png"))

    pca_scatter(tfidf_l, labels, "TF-IDF — PCA", os.path.join(cfg.save_dir, "pca_tfidf.png"))
    pca_scatter(cbow_l,  labels, "CBOW — PCA",   os.path.join(cfg.save_dir, "pca_cbow.png"))
    pca_scatter(skip_l,  labels, "Skip-gram — PCA", os.path.join(cfg.save_dir, "pca_skip.png"))
    print(f"[OK] Saved figures to: {cfg.save_dir}")

    # 6) (Optional) Downstream classification
    if cfg.do_clf:
        clf = LogisticRegression(max_iter=1000)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
        def cv_f1(X, y):
            return float(np.mean(cross_val_score(clf, X, y, scoring="f1_macro", cv=cv)))
        f1_tfidf = cv_f1(tfidf_l, labels)
        f1_cbow  = cv_f1(cbow_l,  labels)
        f1_skip  = cv_f1(skip_l,  labels)
        msg = (f"5-fold Macro F1\n"
               f"TF-IDF : {f1_tfidf:.3f}\n"
               f"CBOW   : {f1_cbow:.3f}\n"
               f"Skip   : {f1_skip:.3f}\n")
        with open(os.path.join(cfg.save_dir, "clf_f1.txt"), "w") as f:
            f.write(msg)
        print("\n=== 5-fold Macro F1 (Company classification) ===")
        print(msg)

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
