import os
import re
import base64
import json
import io
from typing import List, Tuple, Dict, Optional

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from transformers import pipeline
from transformers.utils import logging as hf_logging
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Quiet Transformers + tokenizers warnings
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# If you have these utilities, you can still import them. The app no longer *depends* on them for core flow.
try:
    # Attempt to import optional utilities and the company keywords mapping
    from ensemble_cluster import EnsembleClusterRetriever, weak_label_company, COMPANY_KEYWORDS
except Exception:
    # Provide fallbacks if import fails
    EnsembleClusterRetriever = None
    COMPANY_KEYWORDS: Dict[str, List[str]] = {}
    def weak_label_company(x: str) -> str:
        """
        Fallback weak labelling function used if the ensemble utilities are not available.
        This simplistic implementation looks for a handful of known company names or
        abbreviations in the input string and returns a canonical label. Extend
        `patterns` as needed for your dataset.
        """
        x = str(x)
        patterns = {
            r"\bReliance\b|\bRIL\b": "Reliance Industries",
            r"\bTata\s+Motors\b": "Tata Motors",
            r"\bHDFC\s+Bank\b": "HDFC Bank",
        }
        for pat, lab in patterns.items():
            if re.search(pat, x, flags=re.IGNORECASE):
                return lab
        return "Unknown"

# -----------------------------
# Configuration
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# -----------------------------------------------------------------------------
# Alias mapping and edit-distance helpers
# -----------------------------------------------------------------------------

def _norm_text(s: str) -> str:
    """
    Normalize a string by converting to lowercase, replacing non-alphanumeric
    characters with spaces, and collapsing multiple spaces. This is used to
    ensure consistent comparison between user input and alias strings.

    Parameters
    ----------
    s : str
        Input string to normalize.

    Returns
    -------
    str
        Normalized version of the input string.
    """
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _build_alias_lookup(company_keywords: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Build a lookup dictionary mapping normalized alias strings to their
    canonical company name. Each canonical name is also included in the
    dictionary so that exact matches to the proper name are handled.

    Parameters
    ----------
    company_keywords : Dict[str, List[str]]
        Mapping of canonical names to lists of alias strings (incorrect
        names, abbreviations, shortforms).

    Returns
    -------
    Dict[str, str]
        Dictionary mapping each normalized alias to its canonical name.
    """
    lookup: Dict[str, str] = {}
    for canonical, aliases in company_keywords.items():
        # Include canonical name itself
        norm_canon = _norm_text(canonical)
        if norm_canon:
            lookup[norm_canon] = canonical
        # Include each alias
        for alias in aliases:
            norm_alias = _norm_text(alias)
            if norm_alias:
                lookup[norm_alias] = canonical
    return lookup


# Build the alias lookup once at import time. If COMPANY_KEYWORDS is empty,
# the lookup will also be empty.
ALIAS_LOOKUP: Dict[str, str] = _build_alias_lookup(COMPANY_KEYWORDS)


def scan_aliases_in_query(query: str) -> List[str]:
    """
    Identify canonical company names by scanning the user's query for any
    normalized alias strings. This is used when NER fails to recognise
    organisation names. If an alias appears as a substring in the query, we
    collect the corresponding canonical name.

    Parameters
    ----------
    query : str
        The raw user query.

    Returns
    -------
    List[str]
        List of canonical company names inferred from the query.
    """
    norm_q = _norm_text(query)
    found: List[str] = []
    seen: set = set()
    for alias_norm, canonical in ALIAS_LOOKUP.items():
        if alias_norm and alias_norm in norm_q:
            if canonical not in seen:
                seen.add(canonical)
                found.append(canonical)
    return found


def match_company_by_edit_distance(org: str, companies: List[str]) -> Tuple[Optional[str], Optional[int]]:
    """
    For a single extracted organisation string, compute the edit distance (Levenshtein)
    between its normalized form and every alias in `ALIAS_LOOKUP`. Restrict
    consideration to canonical names present in the current dataset (the
    `companies` list). Return the canonical name with the smallest distance.

    Parameters
    ----------
    org : str
        The organisation string extracted from the query (via NER or alias scan).
    companies : List[str]
        List of canonical company names present in the current dataset.

    Returns
    -------
    Tuple[Optional[str], Optional[int]]
        (best_canonical, distance). If no match is found, both values are None.
    """
    norm_org = _norm_text(org)
    if not norm_org:
        return None, None
    best_canon: Optional[str] = None
    best_dist: Optional[int] = None
    # Iterate over alias lookup; find minimal distance to any alias of a company
    for alias_norm, canonical in ALIAS_LOOKUP.items():
        # Consider only canonical names that are in our current companies list
        if canonical not in companies:
            continue
        dist = Levenshtein.distance(norm_org, alias_norm)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_canon = canonical
            # Early exit if exact match (distance = 0)
            if dist == 0:
                break
    return best_canon, best_dist

# -----------------------------
# Styling
# -----------------------------
def load_css():
    st.markdown(
        """
        <style>
  /* Global: force white bg + black text across Streamlit */
  html, body, .stApp {
    background-color: #ffffff !important;
    color: #000000 !important;
  }

  /* Header */
  .main-header {
    text-align: center;
    padding: 2rem 0;
    background: #ffffff;            /* was gradient */
    color: #000000;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    margin-bottom: 2rem;
  }

  /* Stats strip */
  .stats-container {
    display: flex;
    justify-content: space-around;
    margin: 1rem 0;
    padding: 1rem;
    background-color: #ffffff;      /* was #f0f2f6 */
    border: 1px solid #e5e7eb;
    border-radius: 10px;
  }
  .stat-item { text-align: center; padding: 1rem; }
  .stat-number { font-size: 2rem; font-weight: bold; color: #000000; }
  .stat-label  { font-size: 0.9rem; color: #000000; opacity: 0.7; margin-top: 0.5rem; }

  /* Query box */
  .query-box {
    background: #ffffff;
    color: #000000;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    margin: 1rem 0.
  }

  /* NER results */
  .ner-results {
    background-color: #ffffff;      /* was #e8f4fd */
    color: #000000;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #2e86ab; /* keep accent only on the left */
  }

  /* Headlines list */
  .headline-item {
    padding: 0.8rem;
    margin: 0.5rem 0;
    background-color: #ffffff;      /* was #f8f9fa */
    color: #000000;
    border: 1px solid #f0f0f0;
    border-left: 4px solid #2e86ab;
    border-radius: 5px;
  }

  /* Story card */
  .story-container {
    background: #ffffff;
    color: #000000;
    padding: 2rem;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin: 1rem 0;
  }

  /* Matched company banner */
  .matched-company {
    background-color: #ffffff;      /* was green bg */
    color: #000000;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #28a745; /* keep green accent */
    margin: 1rem 0;
    font-weight: bold;
  }
</style>

        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# NER loader + function
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_ner_pipeline():
    return pipeline(
        "token-classification",
        model="dslim/bert-base-NER",
        aggregation_strategy="average",
    )

def ner_tag(query: str, ner_pipe) -> List[str]:
    ents = ner_pipe(query)
    def clean(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()
    orgs = [clean(e.get("word", "")) for e in ents if e.get("entity_group") == "ORG"]
    # dedupe, preserve order
    seen = set()
    out = []
    for o in orgs:
        ol = o.lower()
        if ol and ol not in seen:
            seen.add(ol)
            out.append(o)
    return out

# -----------------------------
# Data + (optional) retriever
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_data_and_retriever(
    csv_path: str,
    n_clusters: int = 30,
    top_clusters: int = 3,
    temperature: float = 0.25,
    random_state: int = 42,
) -> Tuple[object, Dict[str, List[str]], List[str], int, int]:
    """
    Returns: (retriever_or_None, docs_by_company, companies, total_docs, k_used)
    - Builds a simple weak label for companies using `weak_label_company`.
    - If EnsembleClusterRetriever is available, fit it (optional) for future use.
    - Independently returns docs_by_company for fast TF-IDF ranking inside company.
    """
    df = pd.read_csv(csv_path)
    if "headline" not in df.columns:
        raise ValueError("CSV must contain a 'headline' column.")
    df = df.dropna(subset=["headline"]).drop_duplicates(subset=["headline"]).reset_index(drop=True)

    labels_all = df["headline"].astype(str).apply(weak_label_company).tolist()
    mask = [lab != "Unknown" for lab in labels_all]
    docs = df.loc[mask, "headline"].astype(str).tolist()
    labels = [lab for lab, m in zip(labels_all, mask) if m]

    # Group docs by company for quick intra-company ranking
    docs_by_company: Dict[str, List[str]] = {}
    for text, lab in zip(docs, labels):
        docs_by_company.setdefault(lab, []).append(text)

    # Optional: fit ensemble retriever (kept for compatibility)
    retriever = None
    k_used = 0
    if EnsembleClusterRetriever is not None and len(docs) >= 2:
        k_used = min(n_clusters, max(2, len(docs)))
        retriever = EnsembleClusterRetriever(
            n_clusters=k_used,
            top_clusters=top_clusters,
            temperature=temperature,
            minibatch=True,
            random_state=random_state,
        )
        retriever.fit(docs, meta=labels)

    companies = sorted(docs_by_company.keys())
    return retriever, docs_by_company, companies, len(docs), k_used

# -----------------------------
# Company matching (robust to minor spelling)
# -----------------------------
def find_best_company_match(
    extracted_orgs: List[str],
    companies: List[str],
    threshold: int = 85,
) -> str:
    """
    Given a list of organization strings extracted from the user's query and a
    list of available canonical company names, find the most appropriate
    company match. The matching procedure proceeds in stages:

    1. **Edit-distance matching**: For each extracted organization, compute the
       Levenshtein distance to every known alias via `match_company_by_edit_distance`.
       Return the canonical name with the smallest distance across all
       organizations. An exact match (distance=0) is returned immediately.

    2. **Exact and substring matching**: If no match is found via edit distance,
       perform case-insensitive exact and substring matching against the list
       of canonical names.

    3. **Fuzzy fallback**: As a last resort, perform a fuzzy WRatio match
        between the concatenated organisation strings and the canonical names.

    Parameters
    ----------
    extracted_orgs : List[str]
        Organization names detected by NER or alias scanning.
    companies : List[str]
        Canonical company names available in the current dataset.
    threshold : int, optional
        Minimum fuzzy score (0-100) to accept a fuzzy fallback match.

    Returns
    -------
    str
        The matched canonical company name, or an empty string if no match is
        found.
    """
    # Stage 1: Edit-distance matching across aliases
    best_canon: Optional[str] = None
    best_dist: Optional[int] = None
    for org in extracted_orgs:
        canon, dist = match_company_by_edit_distance(org, companies)
        if canon is not None and dist is not None:
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_canon = canon
                # Early exit if exact match (distance 0) returns immediately
                if dist == 0:
                    return best_canon
    if best_canon:
        return best_canon

    # Stage 2: Exact and substring matching on canonical names
    comp_lc = {c.lower(): c for c in companies}
    for org in extracted_orgs:
        o = org.lower()
        if o in comp_lc:
            return comp_lc[o]
        for c in companies:
            cl = c.lower()
            if o in cl or cl in o:
                return c

    # Stage 3: Fuzzy WRatio fallback
    q = ", ".join(extracted_orgs) if extracted_orgs else ""
    if not q:
        return ""
    match = process.extractOne(q, companies, scorer=fuzz.WRatio)
    if match and match[1] >= threshold:
        return match[0]
    return ""

# -----------------------------
# Intra-company TF-IDF ranker 
# -----------------------------
def rank_company_headlines(query: str, company_docs: List[str], top_k: int = 50) -> List[str]:
    if not company_docs:
        return []
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, norm="l2")
        X = vec.fit_transform(company_docs)
        qv = vec.transform([query])
        sims = linear_kernel(qv, X).ravel()
        top_idx = sims.argsort()[::-1][:top_k]
        return [company_docs[i] for i in top_idx]
    except Exception:
        # very small corpora, fallback to first N
        return company_docs[:top_k]

# -----------------------------
# LLM call (Groq)
# -----------------------------
def call_groq_chat_completion(messages: List[dict], model: str = DEFAULT_MODEL, temperature: float = 0.3) -> str:
    if not GROQ_API_KEY:
        return "Please set your GROQ_API_KEY environment variable."
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "temperature": temperature, "messages": messages}
    try:
        r = requests.post(GROQ_CHAT_URL, json=payload, headers=headers, timeout=60)
        if r.status_code != 200:
            return f"API error {r.status_code}: {r.text}"
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Request error: {e}"

# -----------------------------
# Prompt builder
# -----------------------------
def build_messages(company: str, headlines: List[str], original_query: str = "") -> List[dict]:
    context = "\n".join(f"- {h}" for h in headlines)
    query_context = f" (Based on user query: '{original_query}')" if original_query else ""
    return [
        {
            "role": "system",
            "content": (
                "You are a friendly financial storyteller. Write in simple language for laypeople. "
                "Explain any financial or technical jargon clearly, with brief examples if needed. "
                "Structure your response with clear paragraphs and use engaging language."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Create a compelling and coherent story about {company} using these recent headlines.{query_context}\n"
                f"Focus on what happened, why it matters, and explain jargon in plain English.\n"
                f"Make it engaging and informative for general readers.\n\n"
                f"Headlines:\n{context}\n\n"
                "Output: 2-3 short paragraphs with simple explanations and smooth narrative flow."
            ),
        },
    ]

# -----------------------------
# Financial Jargon explainer
# -----------------------------
def build_jargon_messages(term: str) -> List[dict]:
    """
    Construct a chat prompt for explaining a financial term.  The system prompt
    instructs the model to act as a concise finance educator.  The user prompt
    simply asks for an explanation of the supplied term.  This helper is
    separated from the company story builder to allow different tone and
    structure for jargon definitions.

    Parameters
    ----------
    term : str
        The financial term or jargon to explain.

    Returns
    -------
    List[dict]
        A list of message dictionaries to pass to the LLM via
        `call_groq_chat_completion`.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are an expert financial educator. Explain financial terms "
                "and jargon in simple, non‑technical language suitable for a general audience. "
                "Provide a concise definition and, if appropriate, a brief example to aid understanding."
            ),
        },
        {
            "role": "user",
            "content": f"Explain the financial term: {term}",
        },
    ]

def explain_financial_jargon(term: str) -> str:
    """
    Generate an explanation for a financial term using the Groq chat API.  If
    the API key is not set, this returns an informative message.  Otherwise it
    builds an appropriate message payload and forwards it to
    `call_groq_chat_completion`.

    Parameters
    ----------
    term : str
        The financial term to explain.

    Returns
    -------
    str
        The model's explanation or an error string.
    """
    if not term:
        return "Please enter a financial term to explain."
    # Reuse existing API helper; adjust the model and temperature to favour
    # factual, deterministic output
    msgs = build_jargon_messages(term)
    return call_groq_chat_completion(messages=msgs, model=DEFAULT_MODEL, temperature=0.2)

# -----------------------------
# Streamlit App
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="FinClarity", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    load_css()

    # --- Embed logo (logo.jpeg in same folder) next to the title ---
    try:
        with open("logo.jpeg", "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
    except Exception:
        logo_b64 = ""

    if logo_b64:
        st.markdown(
            f"""
            <div class="main-header">
                <div style="display:flex; align-items:center; gap:16px; justify-content:center;">
                    <img src="data:image/jpeg;base64,{logo_b64}" alt="FinClarity Logo" style="height:56px; width:auto;" />
                    <div style="text-align:left">
                        <h1 style="margin:0">FinClarity</h1>
                        <p style="margin:0">Ask about any company in natural language — we extract the company and generate a story</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="main-header";background-color: black;>
                <h1>FinClarity</h1>
                <p>Ask about any financial company in natural language — we extract the company and generate a story</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.spinner("Loading NER model..."):
        ner_pipeline = load_ner_pipeline()

    # Sidebar
    with st.sidebar:
        st.markdown("## Configuration")
        with st.expander("Data Settings", expanded=True):
            csv_path = st.text_input("CSV File Path", value="sensex_finance_news_daily.csv", help="Path to your financial news CSV file")
        with st.expander("Clustering (optional)"):
            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.number_input("Clusters", min_value=2, max_value=100, value=30, help="Number of clusters (if ensemble retriever is available)")
            with col2:
                top_clusters = st.number_input("Top Clusters", min_value=1, max_value=10, value=3, help="Number of top clusters to retrieve from (if used)")
            temperature = st.slider("Cluster Temperature", 0.05, 1.0, 0.25, 0.05, help="Cluster softmax temperature (if used)")
        with st.expander("Generation"):
            top_k = st.slider("Headlines to Use", 5, 50, 20, 1)
        api_status = "Connected" if GROQ_API_KEY else "Not configured"
        st.markdown(f"**Groq API:** {api_status}")
        st.markdown("**NER:** Ready")

    # Load data & optional retriever
    try:
        with st.spinner("Loading data and preparing index..."):
            retriever, docs_by_company, companies, total_docs, k_used = load_data_and_retriever(
                csv_path=csv_path,
                n_clusters=int(n_clusters),
                top_clusters=int(top_clusters),
                temperature=float(temperature),
            )
        if not companies:
            st.error("No labeled companies found. Check your CSV and weak labeling rules.")
            return

        st.markdown(
            f"""
            <div class="stats-container">
                <div class="stat-item"><div class="stat-number">{total_docs:,}</div><div class="stat-label">Total Headlines</div></div>
                <div class="stat-item"><div class="stat-number">{len(companies)}</div><div class="stat-label">Companies Found</div></div>
                <div class="stat-item"><div class="stat-number">{k_used}</div><div class="stat-label">Clusters Created</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Query input for company stories
        st.markdown(
            """
            <div class="query-box">
                <h3>Ask About a Company</h3>
                <p>Examples:</p>
                <ul>
                    <li>"Give me a story about Reliance Industries"</li>
                    <li>"Tell me what's happening with HDFC Bank"</li>
                    <li>"What's the latest news on Tata Motors?"</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        user_query = st.text_area(
            "Your Query:",
            placeholder="e.g., Give me a story about Reliance Industries",
            height=100,
            key="company_query",
        )

        btn_disabled = not (user_query.strip() and GROQ_API_KEY)
        if st.button(
            "Generate Story",
            type="primary",
            use_container_width=True,
            disabled=btn_disabled,
            key="generate_story",
        ):
            with st.spinner("Processing your query..."):
                # Step 1: NER
                st.subheader("Step 1: Extracting Organizations")
                extracted_orgs = ner_tag(user_query, ner_pipeline)
                # If NER does not detect any organization, try alias scanning on the query text
                if not extracted_orgs:
                    alt_orgs = scan_aliases_in_query(user_query)
                    if alt_orgs:
                        extracted_orgs = alt_orgs
                        st.markdown(
                            f"<div class='ner-results'><strong>Organizations inferred:</strong> {', '.join(extracted_orgs)} (from alias scan)</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        # If no organizations or aliases are found, decide whether to treat the query as jargon.
                        q = user_query.strip()
                        # Simple heuristic: short queries (<=5 words) are likely jargon terms; longer queries probably intended for companies
                        if len(q.split()) <= 5:
                            st.info("No company detected; interpreting your query as a financial term.")
                            explanation = explain_financial_jargon(q)
                            explanation_formatted = explanation.replace('\n', '<br><br>')
                            st.markdown(
                                f"""
                                <div class="story-container">
                                    <h2>Definition: {q}</h2>
                                    <div style="line-height: 1.6; font-size: 1.1rem;">{explanation_formatted}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            # Skip the remainder of the company flow
                            return
                        else:
                            st.error("No organizations found in your query. Please include a company name.")
                            return
                else:
                    st.markdown(
                        f"<div class='ner-results'><strong>Organizations detected:</strong> {', '.join(extracted_orgs)}</div>",
                        unsafe_allow_html=True,
                    )

                # Step 2: Match to known companies
                st.subheader("Step 2: Finding Company Match")
                matched_company = find_best_company_match(extracted_orgs, companies)
                if not matched_company:
                    st.error(f"No matching company found for: {', '.join(extracted_orgs)}")
                    st.info("Available examples: " + ", ".join(companies[:10]) + ("..." if len(companies) > 10 else ""))
                    return
                st.markdown(f"<div class='matched-company'>Matched Company: {matched_company}</div>", unsafe_allow_html=True)

                # Step 3: Retrieve headlines (intra-company TF-IDF rank)
                st.subheader("Step 3: Retrieving Headlines")
                company_docs = docs_by_company.get(matched_company, [])
                headlines = rank_company_headlines(user_query, company_docs, top_k=int(top_k))
                if not headlines:
                    st.error(f"No headlines found for {matched_company}.")
                    return
                with st.expander(f"Retrieved Headlines ({len(headlines)}):", expanded=False):
                    for i, h in enumerate(headlines, 1):
                        st.markdown(f"<div class='headline-item'><strong>{i}.</strong> {h}</div>", unsafe_allow_html=True)

                # Step 4: Generate story
                st.subheader("Step 4: Generating Story")
                messages = build_messages(matched_company, headlines, user_query)
                story = call_groq_chat_completion(messages=messages)
                story_formatted = story.replace('\n', '<br><br>')
                st.markdown(
                    f"""
                    <div class="story-container">
                        <h2>Financial Story: {matched_company}</h2>
                        <div style="line-height: 1.6; font-size: 1.1rem;">{story_formatted}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # -----------------------------
                # Voice control: play the story
                # -----------------------------
                st.subheader("Read the Story Aloud")

                # A) Browser TTS (Web Speech API) inside an iframe component, with chunking + controls
                components.html(
                    f"""
                    <div style="margin-top: 0.5rem; padding: 0.75rem; border: 1px solid #e5e7eb; border-radius: 8px;">
                      <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
                        <button id="tts_read">🔊 Read</button>
                        <button id="tts_pause">⏸ Pause</button>
                        <button id="tts_resume">▶️ Resume</button>
                        <button id="tts_stop">⛔ Stop</button>
                        <label style="margin-left:8px;">Speed
                          <input id="tts_rate" type="range" min="0.7" max="1.4" value="1" step="0.1">
                          <span id="rate_val">1</span>
                        </label>
                      </div>
                      <p style="margin-top:0.5rem; color:#555; font-size:0.9rem;">Uses your browser's speech engine. If you hear nothing, try the MP3 player below.</p>
                    </div>

                    <script>
                      const fullText = {json.dumps(story)};
                      const rateSlider = document.getElementById('tts_rate');
                      const rateVal = document.getElementById('rate_val');
                      rateSlider.addEventListener('input', () => rateVal.textContent = rateSlider.value);

                      function splitIntoChunks(text, maxLen=250) {{
                        const chunks = [];
                        let start = 0;
                        while (start < text.length) {{
                          let end = Math.min(start + maxLen, text.length);
                          // Prefer sentence boundary
                          let cut = text.lastIndexOf('.', end);
                          if (cut <= start) cut = end;
                          chunks.push(text.slice(start, cut).trim());
                          start = cut + 1;
                        }}
                        return chunks.filter(c => c.length > 0);
                      }}

                      function speakAll() {{
                        window.speechSynthesis.cancel();
                        const rate = parseFloat(rateSlider.value || '1');
                        const parts = splitIntoChunks(fullText);
                        for (const part of parts) {{
                          const u = new SpeechSynthesisUtterance(part);
                          u.rate = rate;
                          window.speechSynthesis.speak(u);
                        }}
                      }}

                      // Warm up voices (some browsers load async)
                      window.speechSynthesis.getVoices();
                      window.speechSynthesis.onvoiceschanged = () => {{}};

                      document.getElementById('tts_read').onclick = () => speakAll();
                      document.getElementById('tts_pause').onclick = () => window.speechSynthesis.pause();
                      document.getElementById('tts_resume').onclick = () => window.speechSynthesis.resume();
                      document.getElementById('tts_stop').onclick = () => window.speechSynthesis.cancel();
                    </script>
                    """,
                    height=180,
                )

                # B) Server-side MP3 fallback via gTTS (works even if browser TTS is blocked)
                def _build_mp3_bytes(text: str) -> Optional[bytes]:
                    try:
                        from gtts import gTTS
                        buf = io.BytesIO()
                        # gTTS handles long text; keep lang='en'
                        gTTS(text=text, lang='en').write_to_fp(buf)
                        return buf.getvalue()
                    except Exception:
                        return None

                mp3_bytes = _build_mp3_bytes(story)
                if mp3_bytes:
                    st.audio(mp3_bytes, format="audio/mp3", start_time=0)
                    st.download_button(
                        "Download Audio (MP3)",
                        mp3_bytes,
                        file_name=f"{matched_company.lower().replace(' ', '_')}_story.mp3",
                        mime="audio/mpeg",
                    )
                else:
                    st.caption("MP3 fallback unavailable (gTTS not installed or no internet). Browser TTS above should still work after clicking Read.")

                # Download
                st.download_button(
                    label="Download Story",
                    data=(
                        f"# {matched_company} Financial Story\n\n"
                        f"User Query: {user_query}\n"
                        f"Extracted Organizations: {', '.join(extracted_orgs)}\n"
                        f"Matched Company: {matched_company}\n"
                        f"Headlines Used: {len(headlines)}\n\n"
                        f"{story}"
                    ),
                    file_name=f"{matched_company.lower().replace(' ', '_')}_story.txt",
                    mime="text/plain",
                )

        # ----------------------------------------------
        # Financial jargon explanation section
        # ----------------------------------------------
        st.markdown("---")
        st.markdown("## Explain Financial Jargon")
        st.markdown(
            "Enter a financial term or jargon below to receive a concise, plain‑English explanation. "
            "Examples: EPS, EBITDA, equity, IPO, market capitalization, PE ratio."
        )

        jargon_term = st.text_input(
            "Financial term:",
            value="",
            placeholder="e.g., EBITDA",
            key="jargon_term",
        )
        explain_disabled = not (jargon_term.strip() and GROQ_API_KEY)
        if st.button(
            "Explain Term",
            type="secondary",
            use_container_width=True,
            disabled=explain_disabled,
            key="explain_jargon",
        ):
            with st.spinner("Generating explanation..."):
                explanation = explain_financial_jargon(jargon_term.strip())
                explanation_formatted = explanation.replace('\n', '<br><br>')
                st.markdown(
                    f"""
                    <div class="story-container">
                        <h2>Definition: {jargon_term.strip()}</h2>
                        <div style="line-height: 1.6; font-size: 1.1rem;">{explanation_formatted}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    except FileNotFoundError:
        st.error(f"CSV file not found. Check path.")
    except Exception as e:
        st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>Built with Streamlit, Transformers NER, Groq API</p>
            <p><small>Set the <code>GROQ_API_KEY</code> environment variable to enable generation</small></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
