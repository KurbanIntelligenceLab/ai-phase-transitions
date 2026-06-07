"""
Fetch accepted main conference papers from ICML, ICLR, NeurIPS (OpenReview),
ACL (ACL Anthology), and CVPR (CVF Open Access) for 2017-2025, and export to Excel.
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openreview
import pandas as pd
import requests
from bs4 import BeautifulSoup
from openreview.api import OpenReviewClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
YEARS = list(range(2017, 2026))  # 2017 to 2025 inclusive
# OpenReview: (display name, venue prefix)
OPENREVIEW_CONFERENCES = [
    ("ICML", "ICML.cc"),
    ("ICLR", "ICLR.cc"),
    ("NeurIPS", "NeurIPS.cc"),
]
# ACL and CVPR are fetched from Anthology and CVF (see below)
# PMLR (proceedings.mlr.press) volume numbers for ICML by year (fallback when OpenReview missing)
PMLR_ICML_VOLUMES = {
    2017: 70, 2018: 80, 2019: 97, 2020: 119, 2021: 139, 2022: 162,
    2023: 202, 2024: 235, 2025: 267,
}
# NeurIPS from papers.nips.cc (fallback)
NIPS_CC_YEARS = list(range(2017, 2026))  # 2017-2025
OUTPUT_EXCEL = Path(__file__).resolve().parent / "ml_conferences_accepted_papers_2017_2025.xlsx"
BASE_URL_V2 = "https://api2.openreview.net"
BASE_URL_V1 = "https://api.openreview.net"

# Optional: set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD for higher rate limits
USERNAME = os.environ.get("OPENREVIEW_USERNAME", "")
PASSWORD = os.environ.get("OPENREVIEW_PASSWORD", "")

# Keyword extraction: when keywords missing or very short, extract from title+abstract
KEYWORD_MAX_TERMS = 15
KEYWORD_MIN_LEN_TO_ENRICH = 10  # enrich if keywords length < this
# Generic single words to drop (unless part of a longer phrase) for more specific topic names
KEYWORD_STOPLIST = {
    "language", "models", "model", "learning", "neural", "network", "networks",
    "data", "text", "image", "images", "video", "videos", "detection", "segmentation",
    "object", "objects", "diffusion", "generation", "translation", "knowledge",
    "based", "method", "methods", "approach", "approaches", "training", "deep",
    "representation", "representations", "feature", "features", "visual", "vision",
    "natural", "processing", "recognition", "classification", "prediction", "inference",
    "optimization", "attention", "transformer", "transformers", "embedding", "embeddings",
    "3d", "2d", "multi", "single", "large", "small", "new", "using", "used", "different",
    "multiple", "various", "effective", "efficient", "better", "high", "low", "real",
    "state", "art", "task", "tasks", "problem", "problems", "results", "performance",
    "framework", "algorithm", "algorithms", "system", "systems", "information", "structure",
}


def get_content_value(content: dict, key: str, default: str = "") -> str:
    """Get a string value from note content; handle both API 1 and API 2 shapes."""
    if not content or key not in content:
        return default
    val = content[key]
    if isinstance(val, dict) and "value" in val:
        v = val["value"]
        return v if v is not None else default
    if isinstance(val, list):
        return "; ".join(str(x) for x in val) if val else default
    return str(val) if val is not None else default


def get_content_list(content: dict, key: str) -> list:
    """Get a list value from note content (e.g. keywords, authors)."""
    if not content or key not in content:
        return []
    val = content[key]
    if isinstance(val, dict) and "value" in val:
        val = val["value"]
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        return [val] if val.strip() else []
    return []


def paper_row_from_note(note, year: int, venue_id: str, conference: str) -> dict:
    """Build a flat row dict from an OpenReview note (API 1 or API 2 style)."""
    content = getattr(note, "content", None) or {}
    # Normalize content: API 2 uses content with optional 'value' wrapper
    title = get_content_value(content, "title", "")
    abstract = get_content_value(content, "abstract", "")
    tldr = get_content_value(content, "TLDR", "") or get_content_value(content, "tldr", "")
    keywords_list = get_content_list(content, "keywords") or get_content_list(content, "keyword")
    keywords = "; ".join(keywords_list) if keywords_list else ""
    authors_list = get_content_list(content, "authors") or []
    authors = "; ".join(authors_list) if authors_list else ""

    # PDF: can be in content['pdf'] as value or nested
    pdf_val = content.get("pdf") or {}
    if isinstance(pdf_val, dict):
        pdf_url = pdf_val.get("value") or pdf_val.get("url") or ""
    else:
        pdf_url = str(pdf_val) if pdf_val else ""

    forum = getattr(note, "forum", None) or getattr(note, "id", "")
    note_id = getattr(note, "id", "")
    number = getattr(note, "number", "") or ""
    openreview_url = f"https://openreview.net/forum?id={forum}" if forum else ""

    # Extra useful fields (if present)
    track = get_content_value(content, "track", "") or get_content_value(content, "subject areas", "")
    submission_date = ""
    if hasattr(note, "tcdate") and note.tcdate:
        try:
            from datetime import datetime
            submission_date = datetime.fromtimestamp(note.tcdate / 1000.0).strftime("%Y-%m-%d")
        except Exception:
            submission_date = str(note.tcdate)

    paper_type = "workshop" if "/Workshop" in (venue_id or "") else "main"
    return {
        "conference": conference,
        "year": year,
        "venue": venue_id,
        "paper_type": paper_type,
        "title": title,
        "abstract": abstract,
        "TLDR": tldr,
        "keywords": keywords,
        "authors": authors,
        "track_or_subject_areas": track,
        "submission_date": submission_date,
        "openreview_url": openreview_url,
        "pdf_url": pdf_url,
        "openreview_id": note_id,
        "paper_number": number,
    }


def make_row(conference: str, year: int, venue: str, title: str, abstract: str = "",
             tldr: str = "", keywords: str = "", authors: str = "", track: str = "",
             submission_date: str = "", paper_url: str = "", pdf_url: str = "",
             paper_id: str = "", paper_number: str = "", paper_type: str = "main") -> dict:
    """Build a row dict with same keys as OpenReview rows (for ACL, CVPR)."""
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
        "openreview_url": paper_url,  # reused as main paper URL for all sources
        "pdf_url": pdf_url,
        "openreview_id": paper_id,
        "paper_number": paper_number,
    }


# ---------------------------------------------------------------------------
# ICML from PMLR (proceedings.mlr.press) – fallback/supplement when OpenReview missing
# ---------------------------------------------------------------------------
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


def fetch_icml_from_pmlr(max_workers: int = 10) -> list:
    """Fetch ICML papers from Proceedings of ML Research (https://proceedings.mlr.press/).
    Uses parallel abstract fetching to avoid slow sequential requests.
    """
    rows = []
    for year in YEARS:
        vol = PMLR_ICML_VOLUMES.get(year)
        if not vol:
            continue
        url = f"https://proceedings.mlr.press/v{vol}/"
        seen_titles = set()
        try:
            r = requests.get(
                url,
                timeout=(5, 30),
                headers={"User-Agent": "Mozilla/5.0"},
            )
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # Collect paper candidates (title, authors, abs_url) first
            candidates = []
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if f"/v{vol}/" not in href or not href.endswith(".html"):
                    continue
                if "/assets/" in href or "index" in href:
                    continue
                if "Download PDF" in (a.get_text() or ""):
                    continue
                abs_url = href if href.startswith("http") else f"https://proceedings.mlr.press{href}" if href.startswith("/") else f"https://proceedings.mlr.press/v{vol}/{href}"
                title_el = a.find_parent("p") or a.find_parent("div")
                if not title_el:
                    continue
                text = title_el.get_text(separator="\n", strip=True)
                lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
                if not lines:
                    continue
                title = lines[0]
                if len(title) < 5 or "Proceedings of" in title or "PMLR" in title or "Filter" in title:
                    continue
                key = (year, title[:80])
                if key in seen_titles:
                    continue
                seen_titles.add(key)
                authors = ""
                for ln in lines[1:]:
                    if "Proceedings of" in ln or "PMLR" in ln or ln.startswith("["):
                        break
                    authors = ln.split(";")[0].strip() if ";" in ln else ln
                    break
                base = abs_url.split("/")[-1].replace(".html", "")
                pdf_url = f"https://proceedings.mlr.press/v{vol}/{base}/{base}.pdf"
                candidates.append((year, title, authors, abs_url, pdf_url, base))

            # Fetch abstracts in parallel (avoids slow sequential requests that could hang)
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
                    conference="ICML", year=year, venue=f"ICML {year} (PMLR)",
                    title=title, abstract=abstract, authors=authors,
                    paper_url=abs_url, pdf_url=pdf_url, paper_id=base,
                    paper_type="main",
                ))
            if candidates:
                print(f"  [PMLR {year}] {len(candidates)} papers", end=" ", flush=True)
            time.sleep(0.3)
        except Exception as e:
            print(f"  [PMLR ICML {year}] {e}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# NeurIPS from papers.nips.cc – fallback when OpenReview missing
# ---------------------------------------------------------------------------
def fetch_neurips_from_nips_cc() -> list:
    """Fetch NeurIPS/NIPS papers from https://papers.nips.cc/paper_files/paper/{year}."""
    rows = []
    for year in NIPS_CC_YEARS:
        url = f"https://papers.nips.cc/paper_files/paper/{year}"
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if "-Abstract-" not in href or str(year) not in href:
                    continue
                if "Datasets_and_Benchmarks" in href and "Conference" not in href:
                    continue
                paper_url = href if href.startswith("http") else f"https://papers.nips.cc{href}" if href.startswith("/") else f"https://papers.nips.cc/paper_files/paper/{year}/{href}"
                title = (a.get_text() or "").strip()
                if not title or len(title) < 5:
                    continue
                authors = ""
                nxt = a.find_next_sibling()
                if nxt:
                    authors = nxt.get_text(strip=True).lstrip("_").rstrip("_")
                abstract = ""
                try:
                    rr = requests.get(paper_url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
                    rr.raise_for_status()
                    abs_soup = BeautifulSoup(rr.text, "html.parser")
                    for h in abs_soup.find_all(["h4", "h3", "h2"]):
                        if "abstract" in (h.get_text(strip=True) or "").lower():
                            nxt = h.find_next_sibling()
                            if nxt:
                                abstract = nxt.get_text(separator=" ", strip=True)
                            break
                    if not abstract:
                        for div in abs_soup.find_all("div", class_=re.compile("abstract", re.I)):
                            abstract = div.get_text(separator=" ", strip=True)
                            if len(abstract) > 30:
                                break
                except Exception:
                    pass
                paper_id = paper_url.split("/")[-1].replace("-Abstract-Conference.html", "").replace("-Abstract-Datasets_and_Benchmarks_Track.html", "")
                pdf_url = paper_url.replace("-Abstract-Conference.html", ".pdf").replace("-Abstract-Datasets_and_Benchmarks_Track.html", ".pdf")
                if not pdf_url.endswith(".pdf"):
                    pdf_url = paper_url.rsplit("/", 1)[0] + "/" + paper_id + ".pdf"
                rows.append(make_row(
                    conference="NeurIPS", year=year, venue=f"NeurIPS {year} (papers.nips.cc)",
                    title=title, abstract=abstract, authors=authors,
                    paper_url=paper_url, pdf_url=pdf_url, paper_id=paper_id,
                    paper_type="main",
                ))
                time.sleep(0.1)
            time.sleep(0.3)
        except Exception as e:
            print(f"  [NeurIPS nips.cc {year}] {e}")
    return rows


# ---------------------------------------------------------------------------
# ACL (ACL Anthology) – https://aclanthology.org
# ---------------------------------------------------------------------------
def fetch_acl_papers() -> list:
    """Fetch ACL main conference papers from ACL Anthology (2017-2025)."""
    rows = []
    try:
        from acl_anthology import Anthology
        anthology = Anthology.from_repo()
        papers_dict = getattr(anthology, "papers", None)
        if papers_dict is None:
            raise AttributeError("no papers")
        seen = set()
        for paper_id, paper in (papers_dict.items() if hasattr(papers_dict, "items") else []):
            if paper_id in seen:
                continue
            try:
                # Modern ID: 2020.acl-main.1, 2020.acl-long.2; old: P18-1001, P17-1001
                pid = str(paper_id)
                year = None
                if "." in pid:
                    part = pid.split(".")[0]
                    if part.isdigit() and len(part) == 4:
                        year = int(part)
                else:
                    if pid.startswith("P") and len(pid) > 2:
                        y = pid[1:3]
                        if y.isdigit():
                            year = 2000 + int(y)
                if year is None or year < 2017 or year > 2025:
                    continue
                venue_slug = ""
                if "." in pid:
                    venue_slug = pid.split(".")[1].split("-")[0].lower()
                else:
                    continue
                if venue_slug != "acl":
                    continue
                seen.add(paper_id)
                title = str(paper.title) if paper.title else ""
                abstract = (str(paper.abstract) if paper.abstract else "").strip()
                authors_list = getattr(paper, "authors", None) or []
                authors_str = ""
                if authors_list:
                    parts = []
                    for a in authors_list:
                        n = getattr(a, "name", None)
                        if n:
                            last = getattr(n, "last", "") or ""
                            first = getattr(n, "first", "") or ""
                            parts.append(f"{last}, {first}".strip(", "))
                        else:
                            parts.append(str(a))
                    authors_str = "; ".join(parts)
                pdf_url = ""
                if getattr(paper, "pdf", None):
                    pdf_url = str(paper.pdf) if not hasattr(paper.pdf, "url") else getattr(paper.pdf, "url", "")
                if not pdf_url and hasattr(paper, "url"):
                    pdf_url = getattr(paper, "url", "")
                paper_url = f"https://aclanthology.org/{pid}" if pid else ""
                rows.append(make_row(
                    conference="ACL", year=year, venue=f"ACL Anthology ({year})",
                    title=title, abstract=abstract, authors=authors_str,
                    paper_url=paper_url, pdf_url=pdf_url, paper_id=pid,
                    paper_type="main",
                ))
            except Exception:
                continue
    except ImportError:
        pass
    except Exception as e:
        print(f"  [ACL Anthology] Error: {e}")
    if not rows:
        # Fallback: scrape event pages
        for year in YEARS:
            try:
                url = f"https://aclanthology.org/events/acl-{year}/"
                r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                for span in soup.find_all("span", class_="d-block"):
                    a = span.find("a", href=re.compile(r"^/[A-Z0-9].*"))
                    if not a:
                        continue
                    href = a.get("href", "").strip("/")
                    if not href or "/" in href:
                        continue
                    title = (a.get_text() or "").strip()
                    if not title or len(title) < 5:
                        continue
                    paper_url = f"https://aclanthology.org/{href}"
                    rows.append(make_row(
                        conference="ACL", year=year, venue=f"ACL Anthology ({year})",
                        title=title, paper_url=paper_url, paper_id=href,
                        paper_type="main",
                    ))
                time.sleep(0.5)
            except Exception as e:
                print(f"  [ACL {year}] {e}")
    return rows


def fetch_acl_workshop_papers() -> list:
    """Fetch ACL workshop/SRW/demo papers from ACL Anthology (2017-2025)."""
    # Volume slugs that are workshops, SRW, demos, or tutorials (not main long/short)
    WORKSHOP_SLUGS = ("acl-srw", "acl-demos", "acl-tutorials")
    rows = []
    try:
        from acl_anthology import Anthology
        anthology = Anthology.from_repo()
        papers_dict = getattr(anthology, "papers", None)
        if papers_dict is None:
            raise AttributeError("no papers")
        seen = set()
        for paper_id, paper in (papers_dict.items() if hasattr(papers_dict, "items") else []):
            if paper_id in seen:
                continue
            try:
                pid = str(paper_id)
                year = None
                if "." in pid:
                    part = pid.split(".")[0]
                    if part.isdigit() and len(part) == 4:
                        year = int(part)
                if year is None or year < 2017 or year > 2025:
                    continue
                venue_slug = ""
                if "." in pid:
                    vol_part = pid.split(".")[1]
                    venue_slug = vol_part.split("-")[0].lower() if "-" in vol_part else vol_part.lower()
                    full_slug = vol_part.lower()
                else:
                    continue
                if venue_slug != "acl":
                    continue
                if full_slug in ("acl-main", "acl-long", "acl-short"):
                    continue
                if not any(full_slug.startswith(s) for s in WORKSHOP_SLUGS) and "workshop" not in full_slug:
                    continue
                seen.add(paper_id)
                title = str(paper.title) if paper.title else ""
                abstract = (str(paper.abstract) if paper.abstract else "").strip()
                authors_list = getattr(paper, "authors", None) or []
                authors_str = ""
                if authors_list:
                    parts = []
                    for a in authors_list:
                        n = getattr(a, "name", None)
                        if n:
                            last, first = getattr(n, "last", "") or "", getattr(n, "first", "") or ""
                            parts.append(f"{last}, {first}".strip(", "))
                        else:
                            parts.append(str(a))
                    authors_str = "; ".join(parts)
                pdf_url = getattr(paper, "pdf", None)
                if pdf_url and hasattr(pdf_url, "url"):
                    pdf_url = pdf_url.url
                else:
                    pdf_url = str(pdf_url) if pdf_url else ""
                paper_url = f"https://aclanthology.org/{pid}" if pid else ""
                venue_label = f"ACL Anthology ({year}) Workshop/SRW/Demos"
                rows.append(make_row(
                    conference="ACL", year=year, venue=venue_label,
                    title=title, abstract=abstract, authors=authors_str,
                    paper_url=paper_url, pdf_url=pdf_url, paper_id=pid,
                    paper_type="workshop",
                ))
            except Exception:
                continue
    except ImportError:
        pass
    except Exception as e:
        print(f"  [ACL Workshops] Error: {e}")
    return rows


# ---------------------------------------------------------------------------
# CVPR (CVF Open Access) – https://openaccess.thecvf.com
# Older years (2017–2020) use content_cvpr_YYYY paths; 2021+ use /content/CVPRYYYY/
# 2017 has papers on main page; 2018–2020 require fetching day links first.
# ---------------------------------------------------------------------------
def _cvpr_extract_papers_from_soup(soup, year: int, base_url: str, content_prefix: str, seen_titles: set, rows: list):
    """Extract paper rows from a CVF page. base_url e.g. https://openaccess.thecvf.com/CVPR2017/."""
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
            conference="CVPR", year=year, venue=f"CVPR {year}",
            title=title, authors=authors, paper_url=paper_url,
            pdf_url=pdf_url, paper_id=paper_id,
            paper_type="main",
        ))


def fetch_cvpr_papers() -> list:
    """Fetch CVPR papers from CVF Open Access (2017-2025)."""
    rows = []
    for year in YEARS:
        seen_titles = set()
        base_name = f"CVPR{year}"
        # Older CVF site (2017–2020): content path is content_cvpr_YYYY (underscore, no slash)
        content_prefix_old = f"content_cvpr_{year}"

        if year <= 2020:
            # 2017: papers on main page; 2018–2020: papers on day pages
            if year == 2017:
                url = f"https://openaccess.thecvf.com/{base_name}"
                try:
                    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
                    r.raise_for_status()
                    soup = BeautifulSoup(r.text, "html.parser")
                    base_url = f"https://openaccess.thecvf.com/{base_name}/"
                    _cvpr_extract_papers_from_soup(soup, year, base_url, content_prefix_old, seen_titles, rows)
                except Exception:
                    pass
            else:
                # 2018, 2019, 2020: get day links from main page, then fetch each day
                main_url = f"https://openaccess.thecvf.com/{base_name}.py"
                try:
                    r = requests.get(main_url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
                    r.raise_for_status()
                    soup = BeautifulSoup(r.text, "html.parser")
                    day_links = []
                    for a in soup.find_all("a", href=True):
                        h = a.get("href", "")
                        if f"{base_name}.py?day=" in h or (base_name in h and "day=" in h):
                            day_links.append(h)
                    base_url = f"https://openaccess.thecvf.com/{base_name}.py"
                    for day_href in day_links[:5]:
                        try:
                            if day_href.startswith("http"):
                                day_url = day_href
                            else:
                                day_url = "https://openaccess.thecvf.com/" + day_href.lstrip("/")
                            rr = requests.get(day_url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
                            rr.raise_for_status()
                            day_soup = BeautifulSoup(rr.text, "html.parser")
                            _cvpr_extract_papers_from_soup(day_soup, year, "https://openaccess.thecvf.com/", content_prefix_old, seen_titles, rows)
                            time.sleep(0.2)
                        except Exception:
                            continue
                except Exception:
                    pass
        else:
            # 2021+: modern path /content/CVPRYYYY/
            for base in (f"CVPR{year}", f"CVPR{year}.py"):
                url = f"https://openaccess.thecvf.com/{base}"
                if "?" not in url:
                    url += "?day=all"
                try:
                    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
                    r.raise_for_status()
                    soup = BeautifulSoup(r.text, "html.parser")
                    conf_prefix = base.replace(".py", "")
                    content_prefix_new = f"/content/{conf_prefix}"
                    base_url = "https://openaccess.thecvf.com/"
                    for a in soup.select(f'a[href*="{content_prefix_new}"]'):
                        href = a.get("href", "")
                        if "_paper.html" not in href and "_paper.pdf" not in href:
                            continue
                        title = (a.get_text() or "").strip()
                        if not title or len(title) < 5 or "pdf" in title.lower() or "supp" in title.lower():
                            continue
                        key = (year, title[:100])
                        if key in seen_titles:
                            continue
                        seen_titles.add(key)
                        if "html" in href:
                            paper_url = href if href.startswith("http") else f"https://openaccess.thecvf.com{href}"
                            pdf_url = href.replace("_paper.html", "_paper.pdf").replace("/html/", "/papers/")
                            if not pdf_url.startswith("http"):
                                pdf_url = "https://openaccess.thecvf.com" + pdf_url
                        else:
                            pdf_url = href if href.startswith("http") else f"https://openaccess.thecvf.com{href}"
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
                            conference="CVPR", year=year, venue=f"CVPR {year}",
                            title=title, authors=authors, paper_url=paper_url,
                            pdf_url=pdf_url, paper_id=paper_id,
                            paper_type="main",
                        ))
                    break
                except Exception:
                    continue
        time.sleep(0.3)
    return rows


def fetch_cvpr_workshop_papers() -> list:
    """Fetch CVPR workshop papers from CVF Open Access (2017-2025)."""
    rows = []
    for year in YEARS:
        seen_titles = set()
        base = f"CVPR{year}_workshops"
        menu_url = f"https://openaccess.thecvf.com/{base}/menu"
        workshop_paths = []
        try:
            r = requests.get(menu_url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = (a.get("href", "") or "").strip()
                if base not in href or "/menu" in href:
                    continue
                parts = href.replace("https://openaccess.thecvf.com", "").strip("/").split("/")
                if len(parts) >= 2 and parts[0] == base:
                    path = parts[1]
                    if path and path not in workshop_paths:
                        workshop_paths.append(path)
        except Exception:
            pass
        for ws_path in workshop_paths[:80]:  # limit to avoid too many requests
            try:
                url = f"https://openaccess.thecvf.com/{base}/{ws_path}"
                r = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a.get("href", "")
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
                        href = "https://openaccess.thecvf.com" + (href if href.startswith("/") else "/" + href)
                    if "_paper.html" in href:
                        paper_url = href
                        pdf_url = href.replace("_paper.html", "_paper.pdf").replace("/html/", "/papers/")
                    else:
                        pdf_url = href
                        paper_url = pdf_url
                    paper_id = href.split("/")[-1].replace("_paper.html", "").replace("_paper.pdf", "")
                    authors = ""
                    parent = a.find_parent("p") or a.find_parent("div")
                    if parent:
                        text = parent.get_text(separator=" ", strip=True)
                        if "[" in text:
                            text = text.split("[")[0].strip()
                        parts = text.split(title, 1)
                        if len(parts) > 1:
                            authors = parts[1].strip().strip(",").strip()
                    rows.append(make_row(
                        conference="CVPR", year=year, venue=f"CVPR {year} Workshop",
                        title=title, authors=authors, paper_url=paper_url,
                        pdf_url=pdf_url, paper_id=paper_id,
                        paper_type="workshop",
                    ))
                time.sleep(0.15)
            except Exception:
                continue
        time.sleep(0.3)
    return rows


def get_accepted_papers_api2(
    client: OpenReviewClient, venue_id: str, year: int, conference: str
) -> list:
    """Fetch accepted submissions for an API 2 venue (content.venueid == venue_id)."""
    try:
        notes = list(client.get_all_notes(content={"venueid": venue_id}))
        return [paper_row_from_note(n, year, venue_id, conference) for n in notes]
    except Exception as e:
        print(f"  [API2] Error for {venue_id}: {e}")
        return []


def _is_accept_decision(reply) -> bool:
    """True if reply looks like an Accept decision (API 1 reply dict or note)."""
    content = reply.get("content") if isinstance(reply, dict) else getattr(reply, "content", {}) or {}
    if not content:
        return False
    # Flatten: sometimes decision is in content.decision.value
    def check(val):
        if val is None:
            return False
        if isinstance(val, dict):
            val = val.get("value", val.get("value", ""))
        return bool(val and "Accept" in str(val))
    for key in ("decision", "Decision", "recommendation", "Recommendation"):
        if check(content.get(key)):
            return True
    # Some venues use slightly different key names
    for key in content:
        if key and ("decision" in key.lower() or "recommend" in key.lower()):
            if check(content.get(key)):
                return True
    return False


def _decision_reply_invitation(reply) -> bool:
    """True if reply is from a decision-type invitation (Decision or Meta_Review with recommendation)."""
    inv = reply.get("invitation", "") if isinstance(reply, dict) else getattr(reply, "invitation", "") or ""
    inv = str(inv)
    if "Decision" in inv or "decision" in inv:
        return True
    # Some venues (e.g. ICLR 2019) use Meta_Review with "recommendation" instead of Decision
    if "Meta_Review" in inv:
        content = reply.get("content", {}) if isinstance(reply, dict) else getattr(reply, "content", {}) or {}
        if "recommendation" in content or "Recommendation" in content:
            return True
    return False


def get_accepted_papers_api1(client, venue_id: str, year: int, conference: str) -> list:
    """Fetch accepted submissions for an API 1 venue (Blind_Submission or Submission + Decision/Meta_Review)."""
    def collect_accepted(submissions_list, use_original=True):
        notes = {n.id: n for n in submissions_list}
        accepted = []
        for submission in notes.values():
            replies = getattr(submission, "details", {}) or {}
            direct = replies.get("directReplies") or []
            for reply in direct:
                if not _decision_reply_invitation(reply):
                    continue
                if not _is_accept_decision(reply):
                    continue
                if use_original:
                    orig = replies.get("original")
                    if orig is not None:
                        if isinstance(orig, dict):
                            try:
                                orig_note = openreview.Note.from_json(orig)
                            except Exception:
                                class _Note:
                                    pass
                                orig_note = _Note()
                                orig_note.content = orig.get("content", {})
                                orig_note.forum = orig.get("forum", "")
                                orig_note.id = orig.get("id", "")
                                orig_note.number = orig.get("number", "")
                                orig_note.tcdate = orig.get("tcdate")
                            accepted.append(orig_note)
                        else:
                            accepted.append(orig)
                    else:
                        accepted.append(submission)
                else:
                    accepted.append(submission)
                break
        if not accepted and notes:
            try:
                decision_notes = client.get_all_notes(invitation=f"{venue_id}/-/Decision")
                for d in decision_notes:
                    content = getattr(d, "content", None) or {}
                    dec = content.get("decision") or content.get("Decision") or ""
                    if isinstance(dec, dict):
                        dec = dec.get("value", dec) or ""
                    if dec and "Accept" in str(dec):
                        forum = getattr(d, "forum", None)
                        if forum and forum in notes:
                            accepted.append(notes[forum])
            except Exception:
                pass
        return accepted

    try:
        # Try Blind_Submission first (double-blind)
        submissions = client.get_all_notes(
            invitation=f"{venue_id}/-/Blind_Submission",
            details="directReplies,original",
        )
        accepted_notes = collect_accepted(submissions, use_original=True)

        # Fallback: some years use Submission only (e.g. ICLR 2019 or single-blind)
        if not accepted_notes:
            try:
                submissions = client.get_all_notes(
                    invitation=f"{venue_id}/-/Submission",
                    details="directReplies",
                )
                accepted_notes = collect_accepted(submissions, use_original=False)
            except Exception:
                pass

        return [paper_row_from_note(n, year, venue_id, conference) for n in accepted_notes]
    except Exception as e:
        print(f"  [API1] Error for {venue_id}: {e}")
        return []


# ---------------------------------------------------------------------------
# Fill missing abstracts (CVPR, ACL) by fetching paper page
# ---------------------------------------------------------------------------
def _extract_abstract_from_cvf_page(html: str) -> str:
    """Extract abstract from CVF paper HTML (openaccess.thecvf.com)."""
    soup = BeautifulSoup(html, "html.parser")
    abstract = ""
    for el in soup.find_all(["div", "section"], id=re.compile("abstract", re.I)):
        abstract = el.get_text(separator=" ", strip=True)
        if len(abstract) > 30:
            return abstract
    for el in soup.find_all(["div", "section"], class_=re.compile("abstract", re.I)):
        abstract = el.get_text(separator=" ", strip=True)
        if len(abstract) > 30:
            return abstract
    for h in soup.find_all(["h2", "h3", "h4"]):
        if "abstract" in (h.get_text() or "").lower():
            nxt = h.find_next_sibling()
            if nxt:
                return nxt.get_text(separator=" ", strip=True)
            break
    return abstract


def _extract_abstract_from_acl_page(html: str) -> str:
    """Extract abstract from ACL Anthology paper page."""
    soup = BeautifulSoup(html, "html.parser")
    for div in soup.find_all("div", class_=re.compile("abstract", re.I)):
        t = div.get_text(separator=" ", strip=True)
        if len(t) > 30:
            return t
    for h in soup.find_all(["h2", "h3"]):
        if "abstract" in (h.get_text() or "").lower():
            nxt = h.find_next_sibling()
            if nxt:
                return nxt.get_text(separator=" ", strip=True)
    return ""


def fill_abstracts_cvpr_acl(rows: list) -> None:
    """In-place: fill empty abstract for CVPR/ACL rows by fetching paper URL."""
    for row in rows:
        if row.get("abstract", "").strip():
            continue
        conf = row.get("conference", "")
        if conf not in ("CVPR", "ACL"):
            continue
        url = row.get("openreview_url") or row.get("paper_url") or ""
        if not url or not url.startswith("http"):
            continue
        try:
            r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            if "thecvf.com" in url:
                row["abstract"] = _extract_abstract_from_cvf_page(r.text)
            else:
                row["abstract"] = _extract_abstract_from_acl_page(r.text)
            time.sleep(0.2)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Topic and methods extraction (when keywords missing) – better than KeyBERT for topic/methods
# ---------------------------------------------------------------------------
def extract_topic_and_methods(title: str, abstract: str, max_terms: int = KEYWORD_MAX_TERMS) -> str:
    """
    Extract paper topic and methods from title + abstract (no KeyBERT).
    Returns a string like "Topic: ...; Methods: ..." suitable for keywords column.
    """
    text = f"{title or ''} . {abstract or ''}".strip()
    if len(text) < 20:
        return ""
    text_lower = text.lower()
    # Method cues: sentences that describe what the paper proposes/uses
    method_cues = (
        "we propose", "we present", "we introduce", "we develop", "we use", "we employ",
        "we leverage", "we show", "we demonstrate", "propose a", "present a", "introduce a",
        "framework", "algorithm", "approach", "method", "model", "based on", "leveraging",
        "using ", "employing", "via ", "through ", "by ", "neural ", "deep ", "learning ",
        "transformer", "attention", "reinforcement", "optimization", "embedding",
    )
    sentences = re.split(r"[.!?]\s+", text)
    method_phrases = []
    topic_phrases = []
    # First 1–2 sentences + title for topic
    topic_text = " ".join(sentences[:2]) + " " + (title or "")
    # Words to drop (stopwords + too generic)
    stop = KEYWORD_STOPLIST | {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was",
        "one", "our", "out", "has", "his", "how", "new", "now", "old", "see", "way", "who",
        "did", "get", "its", "let", "put", "say", "she", "too", "use", "with", "this", "from",
        "have", "been", "more", "than", "when", "which", "that",
    }
    def significant_phrases(s, max_len=4):
        words = re.findall(r"[a-z0-9]+", s.lower())
        out = []
        for n in (3, 2, 1):
            for i in range(len(words) - n + 1):
                w = words[i : i + n]
                if any(x in stop for x in w):
                    continue
                phrase = " ".join(w)
                if len(phrase) < 4:
                    continue
                out.append(phrase)
        return out
    for sent in sentences:
        sent_lower = sent.lower()
        if any(cue in sent_lower for cue in method_cues):
            for p in significant_phrases(sent, 3):
                if p not in method_phrases and len(method_phrases) < max_terms // 2:
                    method_phrases.append(p)
    for p in significant_phrases(topic_text, 3):
        if p not in topic_phrases and p not in method_phrases and len(topic_phrases) < max_terms // 2:
            topic_phrases.append(p)
    parts = []
    if topic_phrases:
        parts.append("Topic: " + "; ".join(topic_phrases[:8]))
    if method_phrases:
        parts.append("Methods: " + "; ".join(method_phrases[:8]))
    return " ".join(parts).strip() if parts else ""


# ---------------------------------------------------------------------------
# Keyword extraction (for topic/keyword analysis when keywords missing)
# ---------------------------------------------------------------------------
_KEYBERT_MODEL = None  # loaded once, reused for all papers


def _get_keybert_model():
    """Load KeyBERT model once and reuse (avoids repeated 'LOAD REPORT' messages)."""
    global _KEYBERT_MODEL
    if _KEYBERT_MODEL is not None:
        return _KEYBERT_MODEL
    try:
        import logging
        # Suppress verbose transformer/sentence-transformers loading messages
        for name in ("transformers", "sentence_transformers"):
            log = logging.getLogger(name)
            log.setLevel(logging.WARNING)
        from keybert import KeyBERT
        _KEYBERT_MODEL = KeyBERT()
        return _KEYBERT_MODEL
    except Exception:
        return None


def _filter_generic_keyphrases(phrases: list) -> list:
    """Keep phrases that are specific: drop single-word stoplist terms and subsumed terms."""
    if not phrases:
        return []
    # Prefer longer phrases: sort by word count desc, then by original order
    ordered = sorted(enumerate(phrases), key=lambda x: (-len(x[1].split()), x[0]))
    kept = []
    seen_lower = set()
    for _, phrase in ordered:
        p = phrase.strip()
        if not p:
            continue
        pl = p.lower()
        words = pl.split()
        # Drop if single word and in stoplist
        if len(words) == 1 and pl in KEYWORD_STOPLIST:
            continue
        # Drop if this phrase is a substring of an already kept longer phrase (e.g. drop "language" if we have "language models")
        if any(pl in k and pl != k for k in seen_lower):
            continue
        kept.append(p)
        seen_lower.add(pl)
    return kept


def _extract_keywords_keybert(text: str, max_terms: int = KEYWORD_MAX_TERMS, model=None) -> list:
    """Use KeyBERT for semantic keyphrase extraction. Prefer 2–4 word phrases for specific topic names."""
    if not text or len(text.strip()) < 20:
        return []
    kw_model = model or _get_keybert_model()
    if kw_model is None:
        return []
    try:
        # Prefer multi-word phrases (2–4 words) for meaningful topic names
        try:
            import inspect
            sig = inspect.signature(kw_model.extract_keywords)
            use_mmr = "use_mmr" in sig.parameters
        except Exception:
            use_mmr = False
        kwargs = {
            "keyphrase_ngram_range": (2, 4),
            "stop_words": "english",
            "top_n": max_terms * 2,  # get more candidates, then filter
        }
        if use_mmr:
            kwargs["use_mmr"] = True
            kwargs["diversity"] = 0.75
        if hasattr(kw_model, "extract_keywords"):
            keywords = kw_model.extract_keywords(text, **kwargs)
        else:
            return []
        result = []
        for item in (keywords or []):
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                result.append(str(item[0]).strip())
            elif isinstance(item, str):
                result.append(item.strip())
        result = _filter_generic_keyphrases(result)
        # If we have too few after filtering, allow 1-word phrases that are not in stoplist
        if len(result) < max_terms // 2:
            kwargs_single = {
                "keyphrase_ngram_range": (1, 1),
                "stop_words": "english",
                "top_n": max_terms,
            }
            if use_mmr:
                kwargs_single["use_mmr"] = True
                kwargs_single["diversity"] = 0.7
            single = []
            for item in (kw_model.extract_keywords(text, **kwargs_single) or []):
                if isinstance(item, (list, tuple)) and item:
                    w = str(item[0]).strip().lower()
                    if w not in KEYWORD_STOPLIST and len(w) > 2:
                        single.append(str(item[0]).strip())
            for s in single:
                if s.lower() not in {r.lower() for r in result} and len(result) < max_terms:
                    result.append(s)
        return result[:max_terms]
    except Exception:
        return []


def _extract_keywords_simple(text: str, max_terms: int = KEYWORD_MAX_TERMS) -> list:
    """Fallback: significant phrases (bigrams/trigrams first, then words) for specific topic names."""
    if not text or len(text.strip()) < 10:
        return []
    text = re.sub(r"[^\w\s-]", " ", text.lower())
    words = re.findall(r"[a-z]{3,}", text)
    stop = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was",
        "one", "our", "out", "has", "his", "how", "man", "new", "now", "old", "see", "way",
        "who", "boy", "did", "get", "its", "let", "put", "say", "she", "too", "use",
        "with", "that", "this", "from", "have", "been", "more", "than", "when", "which",
    }
    stop |= KEYWORD_STOPLIST
    # Prefer 2- and 3-gram phrases
    phrase_counts = {}
    for n in (3, 2):
        for i in range(len(words) - n + 1):
            ng = words[i : i + n]
            if any(w in stop for w in ng):
                continue
            phrase = " ".join(ng)
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
    # Single words (not in stop) for filling
    word_counts = {}
    for w in words:
        if w not in stop:
            word_counts[w] = word_counts.get(w, 0) + 1
    # Build result: multi-word phrases first, then single words not in stoplist
    result = []
    for phrase, _ in sorted(phrase_counts.items(), key=lambda x: -x[1])[:max_terms]:
        if phrase not in (p.lower() for p in result):
            result.append(phrase)
    for w, _ in sorted(word_counts.items(), key=lambda x: -x[1]):
        if len(result) >= max_terms:
            break
        if w not in KEYWORD_STOPLIST and w not in (r.lower() for r in result):
            result.append(w)
    return _filter_generic_keyphrases(result)[:max_terms]


def extract_keywords_for_paper(title: str, abstract: str, tldr: str = "", max_terms: int = KEYWORD_MAX_TERMS, model=None) -> str:
    """Extract topic + methods (and optionally KeyBERT) from title + abstract for papers missing keywords."""
    combined = " ".join(filter(None, [title or "", abstract or "", tldr or ""])).strip()
    if len(combined) < 15:
        return ""
    # Prefer topic/methods extraction (paper topic and methods used)
    topic_methods = extract_topic_and_methods(title, abstract, max_terms=max_terms)
    if topic_methods:
        return topic_methods
    keywords = _extract_keywords_keybert(combined, max_terms=max_terms, model=model)
    if not keywords:
        keywords = _extract_keywords_simple(combined, max_terms=max_terms)
    return "; ".join(keywords) if keywords else ""


def enrich_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing or very short keywords using title + abstract (+ TLDR)."""
    if "keywords" not in df.columns or "title" not in df.columns:
        return df
    need = df["keywords"].fillna("").astype(str).str.len() < KEYWORD_MIN_LEN_TO_ENRICH
    if not need.any():
        return df
    n = need.sum()
    print(f"  Enriching keywords for {n} papers (title+abstract)...", end=" ", flush=True)
    # Load KeyBERT once and reuse for all rows (avoids repeated model load messages)
    kw_model = _get_keybert_model()
    for idx in df.index[need]:
        row = df.loc[idx]
        kw = extract_keywords_for_paper(
            row.get("title", ""),
            row.get("abstract", ""),
            row.get("TLDR", ""),
            model=kw_model,
        )
        if kw:
            df.at[idx, "keywords"] = kw
    print("done.", flush=True)
    return df


def main():
    print("ML/NLP/CV Conferences Accepted Papers -> Excel")
    print("OpenReview: ICML, ICLR, NeurIPS | ACL Anthology: ACL | CVF: CVPR")
    print("Years:", YEARS)
    print("Output:", OUTPUT_EXCEL)
    print()

    # API 2 client (required); API 1 client for older venues
    if USERNAME and PASSWORD:
        client_v2 = OpenReviewClient(
            baseurl=BASE_URL_V2, username=USERNAME, password=PASSWORD
        )
    else:
        client_v2 = OpenReviewClient(baseurl=BASE_URL_V2)

    try:
        kwargs = {"baseurl": BASE_URL_V1}
        if USERNAME and PASSWORD:
            kwargs["username"], kwargs["password"] = USERNAME, PASSWORD
        client_v1 = openreview.Client(**kwargs)
    except Exception as e:
        print("API 1 client (optional) not created:", e)
        client_v1 = None

    all_rows = []
    for conference_name, venue_prefix in OPENREVIEW_CONFERENCES:
        for year in YEARS:
            venue_id = f"{venue_prefix}/{year}/Conference"
            print(f"Processing {venue_id} ({conference_name}) ...", end=" ")
            try:
                group = client_v2.get_group(venue_id)
            except Exception:
                # Some older venues (e.g. ICLR 2017) use lowercase "conference"
                try:
                    venue_id = f"{venue_prefix}/{year}/conference"
                    group = client_v2.get_group(venue_id)
                except Exception as e:
                    print(f"skip (venue not found or error: {e})")
                    continue

            is_api2 = getattr(group, "domain", None) is not None
            if is_api2:
                rows = get_accepted_papers_api2(
                    client_v2, venue_id, year, conference_name
                )
            else:
                if client_v1 is None:
                    print("  (skipping: API 1 client unavailable)")
                    rows = []
                else:
                    rows = get_accepted_papers_api1(
                        client_v1, venue_id, year, conference_name
                    )

            print(f" {len(rows)} accepted papers")
            all_rows.extend(rows)
            time.sleep(0.5)  # gentle rate limit

        # OpenReview workshops (same conference/year)
        workshop_venue_id = f"{venue_prefix}/{year}/Workshop"
        try:
            w_group = client_v2.get_group(workshop_venue_id)
            if getattr(w_group, "domain", None) is not None:
                w_rows = get_accepted_papers_api2(client_v2, workshop_venue_id, year, conference_name)
                if w_rows:
                    print(f"  + Workshops: {len(w_rows)} papers")
                    all_rows.extend(w_rows)
            time.sleep(0.3)
        except Exception:
            pass

    # ACL from ACL Anthology (main + workshops)
    print("Fetching ACL (ACL Anthology) main ...", end=" ")
    acl_rows = fetch_acl_papers()
    print(f" {len(acl_rows)} papers")
    all_rows.extend(acl_rows)
    print("Fetching ACL workshops/SRW/demos ...", end=" ")
    acl_ws = fetch_acl_workshop_papers()
    print(f" {len(acl_ws)} papers")
    all_rows.extend(acl_ws)

    # CVPR from CVF Open Access (main + workshops)
    print("Fetching CVPR (CVF Open Access) main ...", end=" ")
    cvpr_rows = fetch_cvpr_papers()
    print(f" {len(cvpr_rows)} papers")
    all_rows.extend(cvpr_rows)
    print("Fetching CVPR workshops ...", end=" ")
    cvpr_ws = fetch_cvpr_workshop_papers()
    print(f" {len(cvpr_ws)} papers")
    all_rows.extend(cvpr_ws)

    # ICML from PMLR (proceedings.mlr.press) for missing/supplement
    print("Fetching ICML from PMLR (proceedings.mlr.press) ...", end=" ")
    pmlr_icml = fetch_icml_from_pmlr()
    print(f" {len(pmlr_icml)} papers")
    # NeurIPS from papers.nips.cc for missing/supplement
    print("Fetching NeurIPS from papers.nips.cc ...", end=" ")
    nips_cc = fetch_neurips_from_nips_cc()
    print(f" {len(nips_cc)} papers")

    # Merge PMLR/NeurIPS: add only if not already present (by conference, year, title)
    def row_key(r):
        return (r.get("conference"), r.get("year"), ((r.get("title") or "")[:100].lower()))
    existing = {row_key(r) for r in all_rows}
    for r in pmlr_icml:
        if row_key(r) not in existing:
            all_rows.append(r)
            existing.add(row_key(r))
    for r in nips_cc:
        if row_key(r) not in existing:
            all_rows.append(r)
            existing.add(row_key(r))

    # Fill missing abstracts for CVPR and ACL (fetch paper page)
    print("Filling missing abstracts (CVPR, ACL) ...", end=" ", flush=True)
    fill_abstracts_cvpr_acl(all_rows)
    print("done.", flush=True)

    if not all_rows:
        print("No papers collected. Exiting.")
        return

    df = pd.DataFrame(all_rows)
    # Ensure paper_type exists (main vs workshop)
    if "paper_type" not in df.columns:
        df["paper_type"] = "main"
    # Reorder columns (conference first, then paper_type)
    col_order = [
        "conference", "year", "venue", "paper_type", "title", "abstract", "TLDR", "keywords",
        "authors", "track_or_subject_areas", "submission_date", "openreview_url",
        "pdf_url", "openreview_id", "paper_number",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    # Enrich missing keywords from title + abstract (for topic/keyword analysis)
    df = enrich_keywords(df)

    # Remove illegal XML/Excel characters (control chars U+0000–U+001F, U+FFFE, U+FFFF, etc.)
    _illegal_xml_re = re.compile(
        "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f\ufeff\ufffe\uffff]"
    )

    def sanitize_for_excel(s):
        if not isinstance(s, str):
            return s
        return _illegal_xml_re.sub(" ", s)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(sanitize_for_excel)

    df.to_excel(OUTPUT_EXCEL, index=False, engine="openpyxl")
    print(f"\nDone. Wrote {len(df)} rows to {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
