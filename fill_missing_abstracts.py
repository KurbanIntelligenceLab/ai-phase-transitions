"""
Fill missing abstracts in the merged Excel workbook by fetching public HTML pages.

- Rows that already have a non-empty abstract are skipped.
- Sources (from `openreview_url` / paper page URL):
  - ACL Anthology (aclanthology.org)
  - CVPR / CVF Open Access (openaccess.thecvf.com)
  - ICML / PMLR (proceedings.mlr.press)
  - NeurIPS (papers.nips.cc, proceedings.neurips.cc)

Optional: Semantic Scholar search-by-title for rows that still have no abstract
after HTML fetch (--semantic-scholar-fallback). Respect their rate limits; optional
SEMANTIC_SCHOLAR_API_KEY env var for higher quotas.

Usage:
  python fill_missing_abstracts.py
  python fill_missing_abstracts.py --input merged_papers_keywords_keybert.xlsx --output merged_papers_keywords_keybert_filled.xlsx
  python fill_missing_abstracts.py --max-workers 6 --semantic-scholar-fallback --s2-delay 3.5
  python fill_missing_abstracts.py --limit 50   # smoke test
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE as _OPENPYXL_ILLEGAL_CHARS
except ImportError:  # pragma: no cover
    _OPENPYXL_ILLEGAL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Non-characters disallowed in XML 1.0 / can break writers
_EXCEL_STRIP_EXTRA = re.compile(r"[\ufffe\uffff]")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "merged_papers_keywords_keybert.xlsx"
DEFAULT_OUTPUT = SCRIPT_DIR / "merged_papers_keywords_keybert_filled.xlsx"

USER_AGENT = "Mozilla/5.0 (compatible; fill-missing-abstracts/1.0; +https://github.com/)"
MIN_ABSTRACT_LEN = 30
REQUEST_TIMEOUT = (6, 25)
MAX_RETRIES = 3
RETRY_SLEEP = 2.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL + text helpers
# ---------------------------------------------------------------------------
def normalize_url(url: str) -> str:
    """Fix common typo https:/host -> https://host."""
    u = (url or "").strip()
    if not u:
        return ""
    if u.startswith("https:/") and not u.startswith("https://"):
        u = "https://" + u[7:].lstrip("/")
    elif u.startswith("http:/") and not u.startswith("http://"):
        u = "http://" + u[6:].lstrip("/")
    return u


def row_has_abstract(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return False
    return len(s) >= MIN_ABSTRACT_LEN


def clean_abstract_text(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if t.lower().startswith("abstract"):
        t = t[8:].lstrip(" :.-").strip()
    return t


def sanitize_excel_str(value: object) -> object:
    """
    Remove characters openpyxl rejects (control chars in scraped text, etc.)
    and enforce Excel's per-cell length limit.
    """
    if not isinstance(value, str):
        return value
    s = _OPENPYXL_ILLEGAL_CHARS.sub("", value)
    s = _EXCEL_STRIP_EXTRA.sub("", s)
    if len(s) > 32767:
        s = s[:32767]
    return s


def sanitize_dataframe_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Clean all string/object columns so to_excel does not raise IllegalCharacterError."""
    out = df.copy()
    for col in out.columns:
        ser = out[col]
        if ser.dtype == object or pd.api.types.is_string_dtype(ser.dtype):
            out[col] = ser.map(lambda x: sanitize_excel_str(x) if isinstance(x, str) else x)
    return out


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def http_get(session: requests.Session, url: str) -> Optional[str]:
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                time.sleep(RETRY_SLEEP * (attempt + 2))
                continue
            r.raise_for_status()
            return r.text
        except Exception as e:
            log.debug("GET fail %s attempt %s: %s", url[:80], attempt + 1, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP)
    return None


# ---------------------------------------------------------------------------
# Venue-specific abstract extraction
# ---------------------------------------------------------------------------
def fetch_abstract_acl(session: requests.Session, url: str) -> str:
    u = normalize_url(url)
    if not u or "aclanthology.org" not in u.lower():
        return ""
    if u.endswith(".pdf"):
        u = u[:-4]
    for candidate in (u, u.rstrip("/") + "/"):
        html = http_get(session, candidate)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        node = soup.select_one("div.card-body.acl-abstract")
        if not node:
            node = soup.find("div", class_=re.compile(r"acl-abstract", re.I))
        if node:
            text = clean_abstract_text(node.get_text(separator=" ", strip=True))
            if len(text) >= MIN_ABSTRACT_LEN:
                return text
    return ""


def fetch_abstract_cvf(session: requests.Session, url: str) -> str:
    u = normalize_url(url)
    if not u or "thecvf.com" not in u.lower():
        return ""
    if u.lower().endswith(".pdf"):
        u = re.sub(r"_paper\.pdf$", "_paper.html", u, flags=re.I)
        u = u.replace("/papers/", "/html/")
    html = http_get(session, u)
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    div = soup.find("div", id="abstract")
    if div:
        text = clean_abstract_text(div.get_text(separator=" ", strip=True))
        if len(text) >= MIN_ABSTRACT_LEN:
            return text
    for d in soup.find_all("div", class_=re.compile("abstract", re.I)):
        text = clean_abstract_text(d.get_text(separator=" ", strip=True))
        if len(text) >= MIN_ABSTRACT_LEN:
            return text
    return ""


def fetch_abstract_pmlr(session: requests.Session, url: str) -> str:
    u = normalize_url(url)
    if not u or "proceedings.mlr.press" not in u.lower():
        return ""
    if not u.endswith(".html"):
        if u.endswith("/"):
            u = u.rstrip("/") + ".html"
        else:
            u = u + ".html"
    html = http_get(session, u)
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for h in soup.find_all(["h4", "h3"]):
        if h.get_text(strip=True).lower() == "abstract":
            nxt = h.find_next_sibling()
            if nxt:
                text = clean_abstract_text(nxt.get_text(separator=" ", strip=True))
                if len(text) >= MIN_ABSTRACT_LEN:
                    return text
            break
    for div in soup.find_all("div", class_=re.compile("abstract", re.I)):
        text = clean_abstract_text(div.get_text(separator=" ", strip=True))
        if len(text) >= MIN_ABSTRACT_LEN:
            return text
    return ""


def fetch_abstract_neurips(session: requests.Session, url: str) -> str:
    u = normalize_url(url)
    if not u or "nips.cc" not in u.lower() and "neurips.cc" not in u.lower():
        return ""
    html = http_get(session, u)
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    p = soup.select_one("p.paper-abstract")
    if p:
        text = clean_abstract_text(p.get_text(separator=" ", strip=True))
        if len(text) >= MIN_ABSTRACT_LEN:
            return text
    for h in soup.find_all(["h4", "h3", "h2"]):
        if "abstract" in (h.get_text(strip=True) or "").lower():
            nxt = h.find_next_sibling()
            if nxt:
                text = clean_abstract_text(nxt.get_text(separator=" ", strip=True))
                if len(text) >= MIN_ABSTRACT_LEN:
                    return text
            break
    for div in soup.find_all("div", class_=re.compile("abstract", re.I)):
        text = clean_abstract_text(div.get_text(separator=" ", strip=True))
        if len(text) >= MIN_ABSTRACT_LEN:
            return text
    return ""


def fetch_abstract_openreview_html(session: requests.Session, url: str) -> str:
    """Best-effort scrape of forum page (may 403 from some networks)."""
    u = normalize_url(url)
    if not u or "openreview.net" not in u.lower():
        return ""
    html = http_get(session, u)
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for sel in ["div.note-content-value", "div.markdown-content"]:
        for node in soup.select(sel):
            t = node.get_text(separator=" ", strip=True)
            if len(t) >= MIN_ABSTRACT_LEN * 2:
                return clean_abstract_text(t)[:8000]
    return ""


def pick_fetcher(url: str) -> Optional[Callable[[requests.Session, str], str]]:
    u = normalize_url(url).lower()
    if "aclanthology.org" in u:
        return fetch_abstract_acl
    if "thecvf.com" in u or "openaccess.thecvf" in u:
        return fetch_abstract_cvf
    if "proceedings.mlr.press" in u:
        return fetch_abstract_pmlr
    if "nips.cc" in u or "neurips.cc" in u or "papers.nips" in u:
        return fetch_abstract_neurips
    if "openreview.net" in u:
        return fetch_abstract_openreview_html
    return None


def fetch_abstract_semantic_scholar(
    session: requests.Session, title: str, api_key: Optional[str] = None, delay: float = 0.0
) -> str:
    """Search by title; returns first hit with a usable abstract."""
    title = (title or "").strip()
    if len(title) < 12:
        return ""
    if delay > 0:
        time.sleep(delay)
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["x-api-key"] = api_key
    q = title[:400]
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": q, "limit": 5, "fields": "title,abstract"}
    try:
        r = session.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 429:
            log.warning("Semantic Scholar rate limited; increase --s2-delay or set SEMANTIC_SCHOLAR_API_KEY")
            return ""
        if r.status_code != 200:
            return ""
        data = r.json()
        for paper in data.get("data") or []:
            ab = (paper.get("abstract") or "").strip()
            if len(ab) >= MIN_ABSTRACT_LEN:
                return ab
    except Exception as e:
        log.debug("S2 error: %s", e)
    return ""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def fill_missing_abstracts(
    df: pd.DataFrame,
    max_workers: int = 8,
    semantic_scholar_fallback: bool = False,
    s2_delay: float = 3.2,
    limit: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Mutates a copy of df: fills `abstract` where missing. Returns (df_out, stats).
    """
    out = df.copy()
    if "abstract" not in out.columns:
        out["abstract"] = ""

    missing_mask = ~out["abstract"].apply(row_has_abstract)
    idxs = out.index[missing_mask].tolist()
    if limit is not None:
        idxs = idxs[: int(limit)]

    stats = {
        "rows_missing_start": int(missing_mask.sum()),
        "rows_to_process": len(idxs),
        "filled_html": 0,
        "filled_s2": 0,
        "still_missing": 0,
    }

    if not idxs:
        log.info("No rows need abstracts.")
        return out, stats

    # Build work list: (index, url, title)
    work = []
    for i in idxs:
        url = out.at[i, "openreview_url"] if "openreview_url" in out.columns else ""
        title = out.at[i, "title"] if "title" in out.columns else ""
        work.append((i, str(url) if pd.notna(url) else "", str(title) if pd.notna(title) else ""))

    def process_one(item: tuple) -> tuple[int, str, str]:
        """Returns (index, abstract, source) source in html|s2|empty."""
        idx, url, title = item
        session = get_session()
        fetcher = pick_fetcher(url)
        ab = ""
        if fetcher:
            ab = fetcher(session, url)
        if ab:
            return idx, ab, "html"
        if semantic_scholar_fallback and title:
            api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
            ab = fetch_abstract_semantic_scholar(session, title, api_key=api_key, delay=s2_delay)
            if ab:
                return idx, ab, "s2"
        return idx, "", "empty"

    log.info("Fetching up to %s abstracts with %s workers...", len(work), max_workers)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one, w): w[0] for w in work}
        for fut in as_completed(futures):
            idx, ab, src = fut.result()
            if ab:
                out.at[idx, "abstract"] = ab
                if src == "html":
                    stats["filled_html"] += 1
                else:
                    stats["filled_s2"] += 1
            done += 1
            if done % 200 == 0 or done == len(work):
                log.info("Progress %s / %s (html=%s, s2=%s)", done, len(work), stats["filled_html"], stats["filled_s2"])

    still = ~out.loc[idxs, "abstract"].apply(row_has_abstract)
    stats["still_missing"] = int(still.sum())
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill missing abstracts in merged papers Excel.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input .xlsx")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output .xlsx")
    ap.add_argument("--max-workers", type=int, default=8, help="Parallel HTTP workers")
    ap.add_argument(
        "--semantic-scholar-fallback",
        action="store_true",
        help="After HTML fetch fails, search Semantic Scholar by title (slow; respect rate limits)",
    )
    ap.add_argument("--s2-delay", type=float, default=3.2, help="Seconds to sleep before each S2 request")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N missing rows (debug)")
    ap.add_argument("--in-place", action="store_true", help="Overwrite --input (same as output=input)")
    args = ap.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    out_path = inp if args.in_place else args.output.expanduser().resolve()

    log.info("Reading %s ...", inp)
    df = pd.read_excel(inp, engine="openpyxl")

    filled, stats = fill_missing_abstracts(
        df,
        max_workers=max(1, args.max_workers),
        semantic_scholar_fallback=args.semantic_scholar_fallback,
        s2_delay=args.s2_delay,
        limit=args.limit,
    )

    log.info(
        "Done. missing_start=%s processed=%s filled_html=%s filled_s2=%s still_missing=%s",
        stats["rows_missing_start"],
        stats["rows_to_process"],
        stats["filled_html"],
        stats["filled_s2"],
        stats["still_missing"],
    )

    log.info("Sanitizing strings for Excel (removing illegal control characters)...")
    filled = sanitize_dataframe_for_excel(filled)
    filled.to_excel(out_path, index=False, engine="openpyxl")
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
