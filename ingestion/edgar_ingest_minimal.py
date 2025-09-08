import os, time, re
from typing import List
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.services.vector_service import ensure_collection, upsert_points

load_dotenv()

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "YourApp ([email protected])")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik_nozeros}/{acc_nodash}/{primary}"

SLEEP = 0.3

def _client():
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=OPENAI_API_KEY)

def resolve_cik(ticker: str) -> str:
    r = requests.get(SEC_TICKER_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    for _, v in data.items():
        if v.get("ticker", "").upper() == ticker.upper():
            return str(v["cik_str"]).zfill(10)
    raise ValueError(f"Ticker not found: {ticker}")

def recent_filings(cik: str, forms: List[str], limit: int = 2):
    url = SEC_SUBMISSIONS.format(cik=cik)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    payload = r.json().get("filings", {}).get("recent", {})
    res = []
    for i, form in enumerate(payload.get("form", [])):
        if form in forms:
            acc = payload["accessionNumber"][i]
            primary = payload["primaryDocument"][i]
            date = payload["filingDate"][i]
            res.append({"form": form, "accession": acc, "primary": primary, "filing_date": date})
        if len(res) >= limit:
            break
    return res

def fetch_doc(cik: str, accession: str, primary: str) -> str:
    acc_nodash = accession.replace("-", "")
    cik_nozeros = str(int(cik))
    url = SEC_ARCHIVES.format(cik_nozeros=cik_nozeros, acc_nodash=acc_nodash, primary=primary)
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.text

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += (chunk_size - overlap)
    return chunks

def embed(texts: List[str]) -> List[List[float]]:
    client = _client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def ingest_ticker(ticker: str, forms: List[str] = ["10-K", "10-Q"], limit: int = 2):
    cik = resolve_cik(ticker)
    filings = recent_filings(cik, forms=forms, limit=limit)
    if not filings:
        print("No filings found"); return
    ensure_collection()
    for f in filings:
        time.sleep(SLEEP)
        html = fetch_doc(cik, f["accession"], f["primary"])
        text = html_to_text(html)
        chunks = chunk_text(text)
        embeddings = embed(chunks)
        points = []
        for i, vec in enumerate(embeddings):
            payload = {
                "doc_id": f["accession"],
                "source_type": f["form"],
                "section": "document",
                "filing_date": f["filing_date"],
                "source_url": f"https://www.sec.gov/ixviewer/doc?action=display&source=Archives&file=/Archives/edgar/data/{int(cik)}/{f['accession'].replace('-', '')}/{f['primary']}",
                "ticker": ticker.upper(),
                "chunk_index": i,
            }
            points.append({"vector": vec, "payload": payload})
        upsert_points(points)
        print(f"Ingested {f['form']} {f['accession']} with {len(points)} chunks.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, help="Ticker symbol, e.g., AAPL")
    ap.add_argument("--limit", type=int, default=2, help="Number of filings to ingest")
    args = ap.parse_args()
    ingest_ticker(args.ticker, limit=args.limit)

