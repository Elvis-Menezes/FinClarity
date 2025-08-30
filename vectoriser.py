import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sqlite3 

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# --- Download NLTK data (only need to do this once) ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# --- 1. DATA CLEANING & TOKENIZATION ---
print("🧹 1. Loading and cleaning data...")
df = pd.read_csv('sensex_finance_news_daily.csv')
df.dropna(subset=['headline'], inplace=True)
df.drop_duplicates(subset=['headline'], inplace=True)
df.reset_index(drop=True, inplace=True)

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Cleans and tokenizes text for NLP."""
    if not isinstance(text, str):
        return []
    # Lowercase, remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

# Apply cleaning and store tokenized headlines
df['tokens'] = df['headline'].apply(preprocess_text)
# Filter out any headlines that became empty after cleaning
df = df[df['tokens'].apply(len) > 0]
tokenized_headlines = df['tokens'].tolist()

# --- 2. VECTORIZATION ---
print("\n2. Vectorizing headlines with 3 different methods...")

# Method A: TF-IDF
print("   -> Method A: TF-IDF")
cleaned_headlines_str = [' '.join(tokens) for tokens in tokenized_headlines]
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_vectors = tfidf_vectorizer.fit_transform(cleaned_headlines_str).toarray()

# Method B & C: Word2Vec (CBOW and Skip-gram)
print("   -> Method B & C: Word2Vec (CBOW & Skip-gram)")
cbow_model = Word2Vec(sentences=tokenized_headlines, vector_size=100, window=5, min_count=1, workers=4, sg=0, epochs=10)
skipgram_model = Word2Vec(sentences=tokenized_headlines, vector_size=100, window=5, min_count=1, workers=4, sg=1, epochs=10)

def get_headline_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

cbow_vectors = np.array([get_headline_vector(tokens, cbow_model) for tokens in tokenized_headlines])
skipgram_vectors = np.array([get_headline_vector(tokens, skipgram_model) for tokens in tokenized_headlines])

# --- 3. EVALUATION & COMPARISON ---
print("\n3. Comparing models using semantic search...")
# (This part remains unchanged as it happens in memory before storage)

query_headline = "Reliance"
query_tokens = preprocess_text(query_headline)

def find_similar_headlines(query_vec, all_vectors, headlines_df, top_n=5):
    similarities = cosine_similarity(query_vec.reshape(1, -1), all_vectors)
    top_indices = np.argsort(similarities[0])[-top_n:][::-1]
    return headlines_df['headline'].iloc[top_indices]

# Evaluate models...
query_vec_tfidf = tfidf_vectorizer.transform([' '.join(query_tokens)]).toarray()
similar_tfidf = find_similar_headlines(query_vec_tfidf, tfidf_vectors, df)

query_vec_cbow = get_headline_vector(query_tokens, cbow_model)
similar_cbow = find_similar_headlines(query_vec_cbow, cbow_vectors, df)

query_vec_skipgram = get_headline_vector(query_tokens, skipgram_model)
similar_skipgram = find_similar_headlines(query_vec_skipgram, skipgram_vectors, df)

print(f"\n--- Query: '{query_headline}' ---")
print("\n**Results from TF-IDF:**")
for headline in similar_tfidf: print(f"  - {headline}")

print("\n**Results from CBOW:**")
for headline in similar_cbow: print(f"  - {headline}")

print("\n**Results from Skip-gram:**")
for headline in similar_skipgram: print(f"  - {headline}")


# --- 4. STORING THE BEST VECTORS IN SQLITE ---
print("\n4. Storing the best vectors (Skip-gram) in SQLite...")
DB_FILE = "news_vectors.db"

# Connect to the SQLite database (it will be created if it doesn't exist)
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Create a table to store headlines and their vectors
# We use BLOB data type to store the serialized numpy array
cursor.execute("""
    CREATE TABLE IF NOT EXISTS news_headlines (
        id INTEGER PRIMARY KEY,
        headline TEXT NOT NULL,
        embedding BLOB NOT NULL
    )
""")

# Clear existing data to avoid duplicates on re-runs
cursor.execute("DELETE FROM news_headlines")

# Insert the data
# We assume Skip-gram is the best model from the comparison above
for index, row in df.iterrows():
    headline_text = row['headline']
    # Convert the numpy array to bytes (BLOB)
    vector_blob = skipgram_vectors[df.index.get_loc(index)].tobytes()
    
    cursor.execute(
        "INSERT INTO news_headlines (headline, embedding) VALUES (?, ?)",
        (headline_text, vector_blob)
    )

# Commit changes and close the connection
conn.commit()
conn.close()

print(f"\nSuccessfully stored headlines and vectors in '{DB_FILE}'.")


# --- 5. EXAMPLE: QUERYING DATA FROM SQLITE ---
print("\n5. Example: Retrieving data and running a search...")
# --- 5. EXAMPLE: QUERYING DATA FROM SQLITE ---
print("\n🔍 5. Example: Retrieving data and running a search...")

def search_from_db(query_headline):
    # Re-connect to the database
    conn_read = sqlite3.connect(DB_FILE)
    cursor_read = conn_read.cursor()

    # Fetch all records from the database
    cursor_read.execute("SELECT headline, embedding FROM news_headlines")
    all_data = cursor_read.fetchall()
    conn_read.close()

    db_headlines = [row[0] for row in all_data]
    
    # --- THIS IS THE CORRECTED LINE ---
    # The vector dtype must be 'float32' to match the Word2Vec model's output
    db_vectors = np.array([np.frombuffer(row[1], dtype=np.float32) for row in all_data])

    # Vectorize the query using the same trained model
    query_tokens = preprocess_text(query_headline)
    query_vector = get_headline_vector(query_tokens, skipgram_model).reshape(1, -1)

    # Calculate similarities (now the dimensions will match)
    similarities = cosine_similarity(query_vector, db_vectors)
    top_indices = np.argsort(similarities[0])[-5:][::-1]
    
    print(f"\n--- Top 5 results from database for query: '{query_headline}' ---")
    for i in top_indices:
        print(f"  - {db_headlines[i]}")

# Run the example search
search_from_db("Sensex and Nifty trade higher")

# ===========================================================
# EVALUATION & VISUALIZATION: TF-IDF vs CBOW vs Skip-gram
# ===========================================================
import re, math, random



# ---------- 0) Weak labels from company keywords ----------
company_keywords = {
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
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["clean_text"] = df["headline"].astype(str).apply(_norm)

def label_company(clean_text: str) -> str:
    best = None; best_len = -1
    for comp, kws in company_keywords.items():
        for kw in kws:
            k = _norm(kw)
            if k and k in clean_text:
                if len(k) > best_len:
                    best, best_len = comp, len(k)
    return best if best else "Unknown"

df["company"] = df["clean_text"].apply(label_company)
labeled_df = df[df["company"] != "Unknown"].copy()

# Use the *same* tokens you already built earlier
# If you used df['tokens'] earlier, re-use; else:
if "tokens" not in labeled_df.columns:
    labeled_df["tokens"] = labeled_df["clean_text"].apply(preprocess_text)

# Align vectors to labeled subset
mask = df.index.isin(labeled_df.index)
tfidf_l = tfidf_vectors[mask]            # (n_labeled, d_tfidf)
cbow_l  = cbow_vectors[mask]
skip_l  = skipgram_vectors[mask]
labels  = labeled_df["company"].values
tokens_l = labeled_df["tokens"].tolist()

# ---------- 1) Retrieval metrics: P@5, nDCG@10, MAP@10 ----------
def precision_at_k(relevant_idx, ranked_idx, k):
    if k == 0: return 0.0
    topk = ranked_idx[:k]
    return len(set(topk) & set(relevant_idx)) / k

def average_precision_at_k(relevant_idx, ranked_idx, k):
    hits, score = 0, 0.0
    for i, idx in enumerate(ranked_idx[:k], start=1):
        if idx in relevant_idx:
            hits += 1
            score += hits / i
    return score / max(1, len(set(relevant_idx)))

def ndcg_at_k(relevant_idx, ranked_idx, k):
    rel_set = set(relevant_idx)
    gains = [1.0 if idx in rel_set else 0.0 for idx in ranked_idx[:k]]
    dcg = sum(g / math.log2(i+2) for i, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i+2) for i, g in enumerate(ideal))
    return (dcg / idcg) if idcg > 0 else 0.0

def rank_all(query_vec, matrix):
    sims = cosine_similarity(query_vec.reshape(1, -1), matrix)[0]
    return np.argsort(-sims)

def query_vec(model_name, company_name):
    q_tok = preprocess_text(company_name)
    if model_name == "TFIDF":
        return tfidf_vectorizer.transform([" ".join(q_tok)]).toarray()[0]
    elif model_name == "CBOW":
        return get_headline_vector(q_tok, cbow_model)
    elif model_name == "SKIP":
        return get_headline_vector(q_tok, skipgram_model)
    else:
        raise ValueError("model_name must be TFIDF/CBOW/SKIP")

embeds = {"TFIDF": tfidf_l, "CBOW": cbow_l, "SKIP": skip_l}
k_list = [5, 10]

rows = []
for comp in sorted(np.unique(labels)):
    rel_idx = np.where(labels == comp)[0]
    if len(rel_idx) < 5:     # skip tiny groups
        continue
    for name, mat in embeds.items():
        qv = query_vec(name, comp)
        ranked = rank_all(qv, mat)
        row = {"company": comp, "model": name}
        row["P@5"]     = precision_at_k(rel_idx, ranked, 5)
        row["MAP@10"]  = average_precision_at_k(rel_idx, ranked, 10)
        row["nDCG@10"] = ndcg_at_k(rel_idx, ranked, 10)
        rows.append(row)

retrieval_df = pd.DataFrame(rows)
macro = retrieval_df.groupby("model", as_index=False)[["P@5", "MAP@10", "nDCG@10"]].mean().round(4)
print("\n=== Retrieval (macro over companies) ===")
print(macro)

# ---------- 2) Visualize: bar charts ----------
def plot_bar(metric):
    plt.figure()
    x = macro["model"].tolist()
    y = macro[metric].tolist()
    plt.bar(x, y)
    plt.title(f"Model comparison — {metric}")
    plt.xlabel("Model"); plt.ylabel(metric)
    plt.tight_layout(); plt.show()

plot_bar("P@5")
plot_bar("nDCG@10")
plot_bar("MAP@10")

# ---------- 3) Visualize: PCA scatter per embedding ----------
def pca_scatter(X, y, title, max_points=2000):
    n = X.shape[0]
    idx = np.arange(n)
    if n > max_points:
        idx = np.random.choice(idx, size=max_points, replace=False)
    Xs, ys = X[idx], y[idx]

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xs)

    plt.figure()
    # plot up to 12 labels for a readable legend
    unique = pd.Series(ys).unique().tolist()
    random.shuffle(unique)
    for lab in unique[:12]:
        m = (ys == lab)
        plt.scatter(X2[m, 0], X2[m, 1], s=12, alpha=0.7, label=lab)
    plt.title(title); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(markerscale=1, fontsize=8, ncol=2)
    plt.tight_layout(); plt.show()

pca_scatter(tfidf_l, labels, "TF-IDF — PCA")
pca_scatter(cbow_l,  labels, "CBOW — PCA")
pca_scatter(skip_l,  labels, "Skip-gram — PCA")

# ---------- 4) (Optional) Downstream task: company classification ----------
# Train a simple classifier on each embedding and compare macro-F1
def clf_score(X, y):
    # small regularized logistic regression; 5-fold CV
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, scoring="f1_macro", cv=cv)
    return float(np.mean(scores)), float(np.std(scores))

f1_tfidf, sd_tfidf = clf_score(tfidf_l, labels)
f1_cbow,  sd_cbow  = clf_score(cbow_l, labels)
f1_skip,  sd_skip  = clf_score(skip_l, labels)

print("\n=== 5-fold Macro F1 (Company classification) ===")
print(f"TF-IDF  : {f1_tfidf:.3f} ± {sd_tfidf:.3f}")
print(f"CBOW    : {f1_cbow:.3f} ± {sd_cbow:.3f}")
print(f"Skipgram: {f1_skip:.3f} ± {sd_skip:.3f}")
