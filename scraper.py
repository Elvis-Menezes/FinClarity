#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinClarity
====================================
- Pulls finance-only news for FinClarity (index) + Sensex-30 companies using Google News via `gnews`.
- Optionally fetches latest price with yfinance (for logging/console only; NOT written to CSV).
- Appends rows to an existing CSV that has columns: ['date', 'headline', '__text_all__']

Install:
    pip install gnews yfinance pandas python-dateutil schedule tqdm

Run once (append to CSV):
    python sensex_finance_news_daily.py --csv sensex_cleaned_lite_merged.csv --max_results 8

Run every day at 06:30 IST (until process is stopped):
    python sensex_finance_news_daily.py --csv sensex_cleaned_lite_merged.csv --daily 06:30 --max_results 8

Notes:
- We try to keep only finance/market-related news by (a) using a finance-focused query and
  (b) filtering titles/descriptions with positive/negative keyword lists.
- We deduplicate against existing CSV by 'headline' and '__text_all__' exact matches.
- CSV schema preserved: date (YYYY-MM-DD HH:MM:SS), headline (str), __text_all__ (str="title - Source").
"""

import argparse
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple

import pandas as pd

try:
    import schedule
except Exception:
    schedule = None

from dateutil import tz
from gnews import GNews
import yfinance as yf

# ----------------------------
# Sensex 30 + Sensex (from user)
# ----------------------------
SENSEX_COMPANIES: Dict[str, str] = {
    "Reliance Industries": "RELIANCE.BO",
    "HDFC Bank": "HDFCBANK.BO",
    "ICICI Bank": "ICICIBANK.BO",
    "Infosys": "INFY.BO",
    "Tata Consultancy Services": "TCS.BO",
    "Hindustan Unilever": "HINDUNILVR.BO",
    "Bharti Airtel": "BHARTIARTL.BO",
    "State Bank of India": "SBIN.BO",
    "Larsen & Toubro": "LT.BO",
    "Bajaj Finance": "BAJFINANCE.BO",
    "Axis Bank": "AXISBANK.BO",
    "Kotak Mahindra Bank": "KOTAKBANK.BO",
    "Maruti Suzuki India": "MARUTI.BO",
    "Asian Paints": "ASIANPAINT.BO",
    "ITC": "ITC.BO",
    "NTPC": "NTPC.BO",
    "Mahindra & Mahindra": "M&M.BO",
    "Sun Pharmaceutical Industries": "SUNPHARMA.BO",
    "Tata Steel": "TATASTEEL.BO",
    "Power Grid Corporation of India": "POWERGRID.BO",
    "HCL Technologies": "HCLTECH.BO",
    "Titan Company": "TITAN.BO",
    "UltraTech Cement": "ULTRACEMCO.BO",
    "Wipro": "WIPRO.BO",
    "Nestle India": "NESTLEIND.BO",
    "Tata Motors": "TATAMOTORS.BO",
    "JSW Steel": "JSWSTEEL.BO",
    "Tech Mahindra": "TECHM.BO",
    "Bajaj Finserv": "BAJAJFINSV.BO",
    "IndusInd Bank": "INDUSINDBK.BO",
}
SEARCH_TOPICS = {"Sensex": "^BSESN"}
SEARCH_TOPICS.update(SENSEX_COMPANIES)

# ----------------------------
# Finance filtering heuristics
# ----------------------------
POSITIVE_RE = re.compile(
    r"(stock|stocks|share|shares|market|markets|bse|nse|sensex|nifty|results|earnings|"
    r"revenue|profit|losses|guidance|dividend|quarter|q1|q2|q3|q4|ipo|fpo|buyback|"
    r"acquisition|merger|deal|stake|valuation|promoter|pledge|rating|brokerage|"
    r"target|downgrade|upgrade|fundraise|rights issue|bonus|split|listing|rbi|sebi)",
    re.IGNORECASE
)

NEGATIVE_RE = re.compile(
    r"(cricket|ipl|football|soccer|tennis|badminton|hockey|kabaddi|olympic|medal|"
    r"match|tournament|cup|series|coach|captain|vs\s|vs\.|defeats|beats)",
    re.IGNORECASE
)

def is_finance_like(title: str, desc: str = "") -> bool:
    text = f"{title} {desc}".strip()
    if NEGATIVE_RE.search(text):
        return False
    return bool(POSITIVE_RE.search(text))

def build_finance_query(company_name: str) -> str:
    # Bias query to business context and exclude sports terms
    negatives = "-cricket -football -tennis -IPL -match -tournament -cup"
    positives = "(stock OR shares OR NSE OR BSE OR results OR earnings OR profit OR revenue OR dividend OR Q1 OR Q2 OR Q3 OR Q4 OR market OR Sensex OR Nifty OR SEBI OR RBI)"
    return f'"{company_name}" {positives} {negatives}'

# ----------------------------
# Core scrape logic
# ----------------------------
def fetch_topic_news(google_news: GNews, topic: str, ticker: str, max_results: int) -> List[Dict[str, Any]]:
    q = build_finance_query(topic)
    google_news.max_results = max_results
    articles = google_news.get_news(q) or []

    rows = []
    for a in articles:
        title = a.get("title", "").strip()
        publisher = (a.get("publisher") or {}).get("title", "").strip()
        desc = a.get("description", "").strip()
        url = a.get("url", "")

        # Filter for finance-only
        if not is_finance_like(title, desc):
            continue

        # Published date -> Asia/Kolkata
        dt = a.get("published date")
        dt_str = ""
        if isinstance(dt, datetime):
            ist = tz.gettz("Asia/Kolkata")
            dt_str = dt.astimezone(ist).strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(dt, str):
            # Best-effort parse
            try:
                # gnews often returns RFC822-like string
                dt_parsed = pd.to_datetime(dt, utc=True, errors="coerce")
                if pd.notna(dt_parsed):
                    dt_parsed = dt_parsed.tz_convert("Asia/Kolkata")
                    dt_str = dt_parsed.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                dt_str = ""

        # CSV schema
        headline = title
        text_all = f"{title} - {publisher}" if publisher else title

        rows.append({
            "date": dt_str,
            "headline": headline,
            "__text_all__": text_all,
            "_meta_source": publisher,
            "_meta_url": url,
            "_meta_ticker": ticker,
            "_meta_topic": topic,
        })
    return rows

def append_to_csv(csv_path: str, new_rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Appends rows to csv_path in the schema (date, headline, __text_all__).
    Deduplicates against existing headlines and __text_all__.
    Returns (added_count, skipped_duplicates).
    """
    cols = ["date", "headline", "__text_all__"]
    try:
        old = pd.read_csv(csv_path)
        if not set(cols).issubset(old.columns):
            print(f"[WARN] CSV at {csv_path} doesn't match expected columns {cols}. I will keep only these three.", file=sys.stderr)
        existing_keys = set(old["headline"].astype(str).str.strip().tolist()) | set(old["__text_all__"].astype(str).str.strip().tolist())
    except FileNotFoundError:
        old = pd.DataFrame(columns=cols)
        existing_keys = set()

    # Prepare new df and drop dups
    new_df = pd.DataFrame(new_rows)
    if new_df.empty:
        return 0, 0

    new_df["headline"] = new_df["headline"].astype(str).str.strip()
    new_df["__text_all__"] = new_df["__text_all__"].astype(str).str.strip()

    mask_new = ~(new_df["headline"].isin(existing_keys) | new_df["__text_all__"].isin(existing_keys))
    deduped = new_df.loc[mask_new, cols].copy()

    skipped = len(new_df) - len(deduped)

    # Ensure date format (string)
    deduped["date"] = deduped["date"].fillna("").astype(str)

    # Append
    out = pd.concat([old[cols], deduped], ignore_index=True)
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return len(deduped), skipped

def run_once(csv_path: str, max_results: int, language: str, country: str) -> None:
    google_news = GNews(language=language, country=country, max_results=max_results)
    ist = tz.gettz("Asia/Kolkata")

    all_rows: List[Dict[str, Any]] = []

    for topic, ticker in SEARCH_TOPICS.items():
        print(f"\n{'='*70}\n{topic} ({ticker})\n{'='*70}")
        # Log price to console (not saved)
        try:
            price = yf.Ticker(ticker).history(period="1d")
            if not price.empty:
                last = float(price["Close"].iloc[-1])
                print(f"Latest Price: ₹{last:.2f}")
        except Exception as e:
            print(f"[WARN] Could not fetch price for {ticker}: {e}")

        rows = fetch_topic_news(google_news, topic, ticker, max_results=max_results)
        print(f"Fetched {len(rows)} finance-filtered articles.")
        all_rows.extend(rows)

        time.sleep(0.5)  # polite

    added, skipped = append_to_csv(csv_path, all_rows)
    print(f"\n[OK] Appended {added} new rows to {csv_path} (skipped {skipped} duplicates).")

def run_daily(csv_path: str, hhmm: str, max_results: int, language: str, country: str) -> None:
    if schedule is None:
        raise RuntimeError("The 'schedule' package is required for --daily. Install with: pip install schedule")

    def job():
        try:
            run_once(csv_path, max_results, language, country)
        except Exception as e:
            print("[ERROR] Daily run failed:", e)

    schedule.every().day.at(hhmm).do(job)
    print(f"[INFO] Scheduled daily job at {hhmm} (local time). Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(1)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finance-only Google News scraper for FinClarity + companies; appends to CSV.")
    p.add_argument("--csv", type=str, default="sensex_cleaned_lite_merged.csv",
                   help="Path to the target CSV with columns [date, headline, __text_all__].")
    p.add_argument("--max_results", type=int, default=8, help="Max articles per topic/company to fetch.")
    p.add_argument("--language", type=str, default="en", help="GNews language, e.g., en")
    p.add_argument("--country", type=str, default="IN", help="GNews country, e.g., IN")
    p.add_argument("--daily", type=str, default=None, metavar="HH:MM",
                   help="If provided (e.g., 06:30), run every day at this time (local). Otherwise runs once and exits.")
    return p.parse_args()

def main():
    args = parse_args()
    if args.daily:
        run_daily(args.csv, args.daily, args.max_results, args.language, args.country)
    else:
        run_once(args.csv, args.max_results, args.language, args.country)

if __name__ == "__main__":
    main()
