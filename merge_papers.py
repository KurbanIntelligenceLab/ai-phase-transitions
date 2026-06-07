"""
Merge CVPR 2019-2020, ICML PMLR, and main ML conferences Excel files into one.
Output: same column format, one combined Excel file.
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILES = [
    BASE_DIR / "cvpr_2019_2020_papers.xlsx",
    BASE_DIR / "ml_conferences_accepted_papers_2017_2025_v6.xlsx",
    BASE_DIR / "neurips_2017_2020_papers.xlsx",
]

# ICML PMLR file (try multiple possible names)
ICML_FILE_OPTIONS = [
    BASE_DIR / "icml_pmlr_papers_v1.xlsx",
    BASE_DIR / "icml_plmr_paper_v1.xlsx",
    BASE_DIR / "icml_pmlr_paper_v1.xlsx",
]

OUTPUT_EXCEL = BASE_DIR / "merged_papers.xlsx"

# Column order for final output (match fetch_icml_papers.py)
COLUMN_ORDER = [
    "conference", "year", "venue", "paper_type", "title", "abstract", "TLDR", "keywords",
    "authors", "track_or_subject_areas", "submission_date", "openreview_url",
    "pdf_url", "openreview_id", "paper_number",
]


def resolve_inputs():
    """Return list of existing input file paths (order: cvpr, icml, ml_conferences)."""
    resolved = []
    for p in INPUT_FILES:
        if p.exists():
            resolved.append(p)
        else:
            print(f"  Warning: not found: {p.name}")
    for p in ICML_FILE_OPTIONS:
        if p.exists():
            resolved.append(p)
            break
    else:
        print("  Warning: no ICML PMLR file found (tried icml_pmlr_papers_v1.xlsx, icml_plmr_paper_v1.xlsx, etc.)")
    return resolved


def load_and_align(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Ensure dataframe has all expected columns; fill missing with empty string."""
    for c in COLUMN_ORDER:
        if c not in df.columns:
            df[c] = ""
    return df[COLUMN_ORDER]


def main():
    print("Merge papers: CVPR 2019-2020 + ICML PMLR + ML conferences 2017-2025")
    paths = resolve_inputs()
    if not paths:
        print("No input files found. Exiting.")
        return

    print(f"Inputs: {[p.name for p in paths]}")
    print(f"Output: {OUTPUT_EXCEL}")
    print()

    frames = []
    for p in paths:
        try:
            df = pd.read_excel(p, engine="openpyxl")
            df = load_and_align(df, p.name)
            frames.append(df)
            print(f"  Loaded {p.name}: {len(df)} rows")
        except Exception as e:
            print(f"  Error reading {p.name}: {e}")

    if not frames:
        print("No data to merge. Exiting.")
        return

    merged = pd.concat(frames, axis=0, ignore_index=True)

    # Drop exact duplicate rows (optional: keep first occurrence)
    before_dedup = len(merged)
    merged = merged.drop_duplicates(keep="first")
    if len(merged) < before_dedup:
        print(f"  Dropped {before_dedup - len(merged)} exact duplicate rows")

    # Optional: drop duplicates by (conference, year, title) to avoid same paper from multiple sources
    key_cols = ["conference", "year", "title"]
    if all(c in merged.columns for c in key_cols):
        before_key = len(merged)
        merged = merged.drop_duplicates(subset=key_cols, keep="first")
        if len(merged) < before_key:
            print(f"  Dropped {before_key - len(merged)} rows duplicate by (conference, year, title)")

    merged.to_excel(OUTPUT_EXCEL, index=False, engine="openpyxl")
    print(f"\nDone. Wrote {len(merged)} rows to {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()
