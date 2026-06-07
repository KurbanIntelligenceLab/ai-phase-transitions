"""
Fetch ICML papers from PMLR (proceedings.mlr.press) volumes only.
Output format matches fetch_icml_papers.py (same columns, Excel export).
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Volume URL -> year (ICML)
PMLR_VOLUMES = [
    ("https://proceedings.mlr.press/v267/", 2025),
    ("https://proceedings.mlr.press/v235/", 2024),
    ("https://proceedings.mlr.press/v202/", 2023),
    ("https://proceedings.mlr.press/v162/", 2022),
    ("https://proceedings.mlr.press/v139/", 2021),
    ("https://proceedings.mlr.press/v119/", 2020),
    ("https://proceedings.mlr.press/v97/", 2019),
    ("https://proceedings.mlr.press/v80/", 2018),
    ("https://proceedings.mlr.press/v70/", 2017),
]

OUTPUT_EXCEL = Path(__file__).resolve().parent / "icml_pmlr_papers.xlsx"
MAX_WORKERS = 10

# Illegal XML/Excel characters to strip
_illegal_xml_re = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f\ufeff\ufffe\uffff]"
)


def make_row(conference: str, year: int, venue: str, title: str, abstract: str = "",
             tldr: str = "", keywords: str = "", authors: str = "", track: str = "",
             submission_date: str = "", paper_url: str = "", pdf_url: str = "",
             paper_id: str = "", paper_number: str = "", paper_type: str = "main") -> dict:
    """Build a row dict with same keys as in fetch_icml_papers.py."""
    return {
        "conference": conference,
        "year": year,
        "venue": venue,
        "paper_type": paper_type,
        "title": title,
        "abstract": abstract,
        "TLDR": tldr,
        "keywords": keywords,
        "authors": authors,
        "track_or_subject_areas": track,
        "submission_date": submission_date,
        "openreview_url": paper_url,
        "pdf_url": pdf_url,
        "openreview_id": paper_id,
        "paper_number": paper_number,
    }


def _fetch_abstract_from_url(abs_url: str, timeout: tuple = (4, 12)) -> str:
    """Fetch and extract abstract from a single PMLR paper URL. Used in parallel."""
    try:
        rr = requests.get(
            abs_url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        rr.raise_for_status()
        abs_soup = BeautifulSoup(rr.text, "html.parser")
        for h in abs_soup.find_all(["h4", "h3"]):
            if h.get_text(strip=True).lower() == "abstract":
                nxt = h.find_next_sibling()
                if nxt:
                    return nxt.get_text(separator=" ", strip=True)
                break
        for div in abs_soup.find_all("div", class_=re.compile("abstract", re.I)):
            abstract = div.get_text(separator=" ", strip=True)
            if len(abstract) > 50:
                return abstract
    except Exception:
        pass
    return ""


def _volume_number_from_url(url: str) -> str:
    """Extract volume number from URL, e.g. v267 -> 267."""
    m = re.search(r"/v(\d+)/", url)
    return m.group(1) if m else ""


def fetch_icml_from_pmlr_volumes(max_workers: int = MAX_WORKERS) -> list:
    """Fetch ICML papers from the configured PMLR volume URLs. Same format as main script."""
    rows = []
    for url, year in PMLR_VOLUMES:
        vol = _volume_number_from_url(url)
        if not vol:
            continue
        seen_titles = set()
        try:
            r = requests.get(
                url,
                timeout=(5, 30),
                headers={"User-Agent": "Mozilla/5.0"},
            )
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            candidates = []
            # PMLR page: paper abstract link has text "abs"; title/authors are in an ancestor block, not the link's immediate parent
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if f"/v{vol}/" not in href or not href.endswith(".html"):
                    continue
                if "/assets/" in href or "index" in href:
                    continue
                if (a.get_text() or "").strip() != "abs":
                    continue
                abs_url = (
                    href if href.startswith("http")
                    else f"https://proceedings.mlr.press{href}" if href.startswith("/")
                    else f"https://proceedings.mlr.press/v{vol}/{href}"
                )
                # Walk up to find the paper block (ancestor with title as first line and authors line with PMLR)
                block = None
                el = a
                for _ in range(8):
                    el = el.find_parent()
                    if el is None:
                        break
                    text = el.get_text(separator="\n", strip=True)
                    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
                    if len(lines) < 2 or len(lines[0]) < 10:
                        continue
                    if "Proceedings of" in lines[0] or "PMLR" in lines[0] or "Filter" in lines[0]:
                        continue
                    if lines[0].startswith("[") or "Download PDF" in lines[0]:
                        continue
                    if not any("PMLR" in ln or "Proceedings of" in ln for ln in lines[1:]):
                        continue
                    block = el
                    break
                if block is None:
                    continue
                text = block.get_text(separator="\n", strip=True)
                lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
                title = lines[0]
                if len(title) < 5 or "Proceedings of" in title or "PMLR" in title or "Filter" in title:
                    continue
                key = (year, title[:80])
                if key in seen_titles:
                    continue
                seen_titles.add(key)
                authors = ""
                for ln in lines[1:]:
                    if "Proceedings of" in ln or "PMLR" in ln:
                        authors = ln.split(";")[0].strip() if ";" in ln else ln
                        break
                    if ln.startswith("["):
                        break
                base = abs_url.split("/")[-1].replace(".html", "")
                pdf_url = f"https://proceedings.mlr.press/v{vol}/{base}/{base}.pdf"
                candidates.append((year, title, authors, abs_url, pdf_url, base))

            # Fetch abstracts in parallel
            abstracts = [""] * len(candidates)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(_fetch_abstract_from_url, c[3]): i
                    for i, c in enumerate(candidates)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        abstracts[idx] = future.result(timeout=20) or ""
                    except Exception:
                        pass

            for (year, title, authors, abs_url, pdf_url, base), abstract in zip(candidates, abstracts):
                rows.append(make_row(
                    conference="ICML",
                    year=year,
                    venue=f"ICML {year} (PMLR)",
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    paper_url=abs_url,
                    pdf_url=pdf_url,
                    paper_id=base,
                    paper_type="main",
                ))
            if candidates:
                print(f"  [PMLR v{vol} / ICML {year}] {len(candidates)} papers", flush=True)
            time.sleep(0.3)
        except Exception as e:
            print(f"  [PMLR v{vol} ICML {year}] {e}", flush=True)
    return rows


def main():
    print("Fetching ICML papers from PMLR volumes only")
    print("Volumes:", [f"v{_volume_number_from_url(u)} ({y})" for u, y in PMLR_VOLUMES])
    print("Output:", OUTPUT_EXCEL)
    print()

    rows = fetch_icml_from_pmlr_volumes()
    if not rows:
        print("No papers collected. Exiting.")
        return

    df = pd.DataFrame(rows)
    col_order = [
        "conference", "year", "venue", "paper_type", "title", "abstract", "TLDR", "keywords",
        "authors", "track_or_subject_areas", "submission_date", "openreview_url",
        "pdf_url", "openreview_id", "paper_number",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    def sanitize(s):
        if not isinstance(s, str):
            return s
        return _illegal_xml_re.sub(" ", s)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(sanitize)

    df.to_excel(OUTPUT_EXCEL, index=False, engine="openpyxl")
    print(f"\nDone. Wrote {len(df)} rows to {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
