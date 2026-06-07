"""
Fetch CVPR 2019 and 2020 papers from CVF Open Access only.
Output format matches fetch_icml_papers.py (same columns, Excel export).
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CVPR_PAGES = [
    ("https://openaccess.thecvf.com/CVPR2019", 2019),
    ("https://openaccess.thecvf.com/CVPR2020", 2020),
]

OUTPUT_EXCEL = Path(__file__).resolve().parent / "cvpr_2019_2020_papers.xlsx"

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


def _extract_papers_from_soup(soup, year: int, base_url: str, content_prefix: str,
                              seen_titles: set, rows: list) -> None:
    """Extract paper rows from a CVF page. base_url e.g. https://openaccess.thecvf.com/CVPR2019/."""
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if content_prefix not in href:
            continue
        if "_paper.html" not in href and "_paper.pdf" not in href:
            continue
        title = (a.get_text() or "").strip()
        if not title or len(title) < 5 or "pdf" in title.lower() or "supp" in title.lower():
            continue
        key = (year, title[:100])
        if key in seen_titles:
            continue
        seen_titles.add(key)
        if not href.startswith("http"):
            href = (base_url.rstrip("/") + "/" + href.lstrip("/")).replace("//", "/")
            if not href.startswith("http"):
                href = "https://openaccess.thecvf.com/" + href.lstrip("/")
        if "_paper.html" in href:
            paper_url = href
            pdf_url = href.replace("_paper.html", "_paper.pdf").replace("/html/", "/papers/")
        else:
            pdf_url = href
            paper_url = pdf_url
        authors = ""
        parent = a.find_parent("p") or a.find_parent("div")
        if parent:
            text = parent.get_text(separator=" ", strip=True)
            if "[" in text:
                text = text.split("[")[0].strip()
            parts = text.split(title, 1)
            if len(parts) > 1:
                authors = parts[1].strip().strip(",").strip()
        paper_id = href.split("/")[-1].replace("_paper.html", "").replace("_paper.pdf", "")
        rows.append(make_row(
            conference="CVPR",
            year=year,
            venue=f"CVPR {year}",
            title=title,
            authors=authors,
            paper_url=paper_url,
            pdf_url=pdf_url,
            paper_id=paper_id,
            paper_type="main",
        ))


def fetch_cvpr_2019_2020() -> list:
    """Fetch CVPR 2019 and 2020 from CVF Open Access. 2019/2020 use day links."""
    rows = []
    for base_url, year in CVPR_PAGES:
        base_url = base_url.rstrip("/")
        base_name = f"CVPR{year}"
        # CVF uses content_CVPR_YYYY (capital CVPR) in hrefs on day pages
        content_prefix = f"content_CVPR_{year}"
        seen_titles = set()

        # Try .py URL for main page (CVF uses this for 2018–2020 to show day links)
        main_url = f"{base_url}.py" if not base_url.endswith(".py") else base_url
        try:
            r = requests.get(main_url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            print(f"  [CVPR {year}] Failed to load main page: {e}", flush=True)
            continue

        day_links = []
        for a in soup.find_all("a", href=True):
            h = a.get("href", "")
            if f"{base_name}.py?day=" in h or (base_name in h and "day=" in h):
                day_links.append(h)

        if day_links:
            for day_href in day_links[:5]:
                try:
                    if day_href.startswith("http"):
                        day_url = day_href
                    else:
                        day_url = "https://openaccess.thecvf.com/" + day_href.lstrip("/")
                    rr = requests.get(day_url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
                    rr.raise_for_status()
                    day_soup = BeautifulSoup(rr.text, "html.parser")
                    _extract_papers_from_soup(
                        day_soup, year, "https://openaccess.thecvf.com/",
                        content_prefix, seen_titles, rows
                    )
                    time.sleep(0.2)
                except Exception:
                    continue
            print(f"  [CVPR {year}] {len([r for r in rows if r['year'] == year])} papers (via day pages)", flush=True)
        else:
            # No day links: try extracting from main page directly (e.g. if structure differs)
            try:
                _extract_papers_from_soup(
                    soup, year, base_url + "/", content_prefix, seen_titles, rows
                )
                n = len([r for r in rows if r["year"] == year])
                print(f"  [CVPR {year}] {n} papers (main page)", flush=True)
            except Exception as e:
                print(f"  [CVPR {year}] {e}", flush=True)
        time.sleep(0.3)

    return rows


def main():
    print("Fetching CVPR 2019–2020 from CVF Open Access")
    print("URLs:", [u for u, _ in CVPR_PAGES])
    print("Output:", OUTPUT_EXCEL)
    print()

    rows = fetch_cvpr_2019_2020()
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
