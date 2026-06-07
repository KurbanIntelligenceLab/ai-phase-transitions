"""
Fetch NeurIPS papers 2017-2020 from papers.nips.cc.
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
NIPS_PAGES = [
    ("https://papers.nips.cc/paper/2017", 2017),
    ("https://papers.nips.cc/paper/2018", 2018),
    ("https://papers.nips.cc/paper/2019", 2019),
    ("https://papers.nips.cc/paper/2020", 2020),
]

OUTPUT_EXCEL = Path(__file__).resolve().parent / "neurips_2017_2020_papers.xlsx"
MAX_WORKERS = 10

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


def _fetch_abstract_from_url(paper_url: str, timeout: tuple = (4, 12)) -> str:
    """Fetch and extract abstract from a single NeurIPS paper URL."""
    try:
        rr = requests.get(
            paper_url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        rr.raise_for_status()
        soup = BeautifulSoup(rr.text, "html.parser")
        for h in soup.find_all(["h4", "h3", "h2"]):
            if "abstract" in (h.get_text(strip=True) or "").lower():
                nxt = h.find_next_sibling()
                if nxt:
                    return nxt.get_text(separator=" ", strip=True)
                break
        for div in soup.find_all("div", class_=re.compile("abstract", re.I)):
            abstract = div.get_text(separator=" ", strip=True)
            if len(abstract) > 30:
                return abstract
    except Exception:
        pass
    return ""


def _paper_url_to_pdf(paper_url: str) -> str:
    """Derive PDF URL from paper abstract page URL."""
    if "-Abstract.html" in paper_url:
        return paper_url.replace("-Abstract.html", ".pdf")
    if "-Abstract-Conference.html" in paper_url:
        return paper_url.replace("-Abstract-Conference.html", ".pdf")
    if "-Abstract-Datasets_and_Benchmarks_Track.html" in paper_url:
        return paper_url.replace("-Abstract-Datasets_and_Benchmarks_Track.html", ".pdf")
    if paper_url.endswith(".html"):
        return paper_url.rsplit("/", 1)[0] + "/" + paper_url.split("/")[-1].replace(".html", ".pdf")
    return paper_url


def _paper_url_to_id(paper_url: str) -> str:
    """Extract paper id from URL (for openreview_id column)."""
    base = paper_url.split("/")[-1]
    for suffix in ("-Abstract.html", "-Abstract-Conference.html", "-Abstract-Datasets_and_Benchmarks_Track.html", ".html"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base.replace(".html", "")


def fetch_neurips_2017_2020(max_workers: int = MAX_WORKERS) -> list:
    """Fetch NeurIPS 2017-2020 from papers.nips.cc list pages."""
    rows = []
    for list_url, year in NIPS_PAGES:
        try:
            r = requests.get(
                list_url,
                timeout=(5, 45),
                headers={"User-Agent": "Mozilla/5.0"},
            )
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            print(f"  [NeurIPS {year}] Failed to load list page: {e}", flush=True)
            continue

        candidates = []
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if "Abstract" not in href or str(year) not in href:
                continue
            if "Datasets_and_Benchmarks" in href and "Conference" not in href:
                continue
            paper_url = (
                href if href.startswith("http")
                else f"https://papers.nips.cc{href}" if href.startswith("/")
                else f"https://papers.nips.cc/{href}" if href.startswith("paper_files")
                else f"https://papers.nips.cc/paper/{year}/{href}"
            )
            title = (a.get_text() or "").strip()
            if not title or len(title) < 5:
                continue
            authors = ""
            parent = a.find_parent()
            if parent:
                full_text = parent.get_text(separator=" ", strip=True)
                if title in full_text:
                    rest = full_text.split(title, 1)[-1].strip()
                    if rest:
                        authors = rest
            if not authors:
                nxt = a.find_next_sibling()
                if nxt:
                    authors = nxt.get_text(strip=True).lstrip("_").rstrip("_")
            paper_id = _paper_url_to_id(paper_url)
            pdf_url = _paper_url_to_pdf(paper_url)
            candidates.append((year, title, authors, paper_url, pdf_url, paper_id))

        # Dedupe by title within year
        seen = set()
        unique = []
        for c in candidates:
            key = (c[0], (c[1] or "")[:100])
            if key in seen:
                continue
            seen.add(key)
            unique.append(c)

        # Fetch abstracts in parallel
        abstracts = [""] * len(unique)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_fetch_abstract_from_url, c[3]): i
                for i, c in enumerate(unique)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    abstracts[idx] = future.result(timeout=20) or ""
                except Exception:
                    pass

        for (year, title, authors, paper_url, pdf_url, paper_id), abstract in zip(unique, abstracts):
            rows.append(make_row(
                conference="NeurIPS",
                year=year,
                venue=f"NeurIPS {year} (papers.nips.cc)",
                title=title,
                abstract=abstract,
                authors=authors,
                paper_url=paper_url,
                pdf_url=pdf_url,
                paper_id=paper_id,
                paper_type="main",
            ))
        print(f"  [NeurIPS {year}] {len(unique)} papers", flush=True)
        time.sleep(0.3)

    return rows


def main():
    print("Fetching NeurIPS 2017-2020 from papers.nips.cc")
    print("URLs:", [u for u, _ in NIPS_PAGES])
    print("Output:", OUTPUT_EXCEL)
    print()

    rows = fetch_neurips_2017_2020()
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
