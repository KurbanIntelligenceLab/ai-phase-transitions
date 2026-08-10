# Topical Phase Transitions in AI Research
### Large-Scale Evidence and an Early-Warning Signature for Emerging Topics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20635335.svg)](https://doi.org/10.5281/zenodo.20635335)
[![arXiv](https://img.shields.io/badge/arXiv-2606.12828-b31b1b.svg)](https://arxiv.org/abs/2606.12828)

> **Paper submitted to:** *Quantitative Science Studies (QSS)*  
> **Title:** "Topical Phase Transitions in Artificial Intelligence Research: Large-Scale Evidence and an Early-Warning Signature for Emerging Topics"

---

## Overview

This repository contains all code and figures supporting the paper. The project tracks **80,814 accepted main-track papers** from five major AI/ML conferences (ACL, CVPR, ICLR, ICML, NeurIPS) over 2017–2025, extracts research topics via KeyBERT, and formalises and validates a **pre-explosion signature** that forecasts which topics will undergo a rapid phase transition (3× growth) within three years.

### Key findings

- AI research volume grew at **29.9% CAGR** (2,445 → 19,800 papers, 2017–2025).
- Large Language Models (LLMs) and diffusion models underwent the most dramatic topical phase transitions post-2022.
- A four-criterion early-warning signature achieves **63% recall** at **27% precision** (vs. 13.5% base rate) in a strict out-of-sample backtest on 223 topics.
- The signature is evaluated at a fixed freeze date (2022) with no peeking at outcome years (2023–2025).

---

## Repository structure

```
paper/
│
├── ── Data collection ──
├── fetch_icml_pmlr_only.py           # Scrape ICML papers from PMLR proceedings
├── fetch_icml_papers.py              # Fetch ICML via OpenReview API (alternative)
├── fetch_cvpr_2019_2020.py           # Scrape CVPR 2019–2020 paper listings
├── fetch_neurips_2017_2020.py        # Scrape NeurIPS 2017–2020 paper listings
├── fill_missing_abstracts.py         # Fill missing abstracts (ACL Anthology, CVF,
│                                     #   PMLR, NeurIPS, Semantic Scholar fallback)
│
├── ── Merging & keyword extraction ──
├── merge_papers.py                   # Merge per-venue Excel files into one dataset
├── create_keywords_from_titles.py    # Simple bag-of-words keywords from titles
├── create_keywords_vocabulary_match.py # Controlled-vocabulary keyword matching
├── create_keywords_keybert_title.py  # KeyBERT keyword extraction from titles
│
├── ── Analysis & figures ──
├── topic_analysis.py                 # Master figure script: counts, heatmaps,
│                                     #   LLM/diffusion/RL trends, word clouds
├── rebuild_figures.py                # Regenerate paper figures from exact data values
├── compute_popularity.py             # Popularity scoring (Xiong et al. 2019 method)
│
├── ── Validation ──
├── backtest_signature.py             # Out-of-sample backtest of the pre-explosion
│                                     #   signature (Section 5 of the paper)
│
├── ── Environment ──
├── environment.yml                   # Conda environment specification
├── requirements.txt                  # pip dependencies (core)
├── requirements-full.txt             # pip dependencies (full: KeyBERT + PyTorch)
│
├── ── Figures (final, committed) ──
├── topic_analysis_figures_keybert/   # All figures used in the paper (PDF + PNG)
│   ├── 01_papers_per_year_by_conference.png
│   ├── 05_top10_topics_per_conference.png
│   ├── 06_heatmap_topics_over_years.png
│   ├── 07_line_top5_topics_over_years.png
│   ├── 09_line_top25_topics_years_*.{png,pdf}  (one per venue)
│   ├── 10_exponential_growth_total_and_stacked.png
│   ├── 12_llm_revolution_over_years.png
│   ├── 14_diffusion_rise_over_years.png
│   ├── 15_rl_over_years_all.png
│   └── summary_papers_per_year_per_conference.csv
│
└── architecture.png                  # System architecture diagram
```

> **Data files** (`.xlsx`, `.csv`) are excluded from Git due to size. The canonical dataset is archived at Zenodo: [doi:10.5281/zenodo.20635335](https://doi.org/10.5281/zenodo.20635335).

---

## Dataset

| Attribute | Value |
|---|---|
| Total papers | **80,814** (main-track only, workshops excluded) |
| Venues | ACL, CVPR, ICLR, ICML, NeurIPS |
| Years | 2017–2025 |
| Keyword method | KeyBERT (`all-MiniLM-L6-v2`), MMR diversity=0.7, n-gram (1,3), top-12 terms |
| Source | Official proceedings (ACL Anthology, CVF, PMLR, OpenReview, NeurIPS.cc) |
| Zenodo archive | [doi:10.5281/zenodo.20635335](https://doi.org/10.5281/zenodo.20635335) |

**Venue breakdown (2025):**

| Venue | Papers (2025) | Share (all years) |
|---|---|---|
| NeurIPS | 5,286 | 27.9% |
| ACL | 4,547 | 21.8% |
| ICLR | 3,703 | 13.9% |
| ICML | 3,402 | 16.3% |
| CVPR | 2,862 | 20.0% |

---

## The pre-explosion signature

The signature flags a topic as likely to explode (≥ 3× growth in 3 years) if **all four** criteria hold at the freeze date:

| Criterion | Threshold |
|---|---|
| **Recency** | First reached ≥ 3 papers/year within the preceding 3 years |
| **Acceleration** | ≥ 2.5× year-on-year growth in at least one of the last two intervals |
| **Cross-venue spread** | Present in ≥ 2 of the 5 venues |
| **Pre-saturation scale** | Annual count between 5 and 300 papers |

**Backtest results** (freeze: 2022, outcome window: 2023–2025, evaluated on 223 topics with ≥ 10 papers in 2022):

| | Exploded | Did not explode |
|---|---|---|
| **Signature positive** | 19 (TP) | 52 (FP) |
| **Signature negative** | 11 (FN) | 141 (TN) |

- Precision: **27%** (vs. 13.5% base rate — 2× lift)
- Recall: **63%**
- No leakage: signature uses only years ≤ 2022; outcomes use only years ≥ 2023.

---

## Setup

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate ml-conference-papers
```

To enable KeyBERT keyword extraction and word clouds, uncomment the `pip:` lines in `environment.yml`, then:

```bash
conda env update -f environment.yml --prune
```

### Option B — pip + venv

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

For the full stack (KeyBERT, PyTorch, word clouds):

```bash
pip install -r requirements-full.txt
```

### Dependencies summary

| Group | Packages |
|---|---|
| Core | `pandas`, `openpyxl`, `matplotlib`, `seaborn`, `requests`, `beautifulsoup4` |
| OpenReview scraping | `openreview-py` |
| KeyBERT keywords | `keybert`, `sentence-transformers` (installs PyTorch) |
| Word clouds | `wordcloud` |

---

## Reproducing the analysis

### Step 1 — Obtain the dataset

Download `merged_papers_keywords_keybert.xlsx` from the Zenodo archive:

```
https://doi.org/10.5281/zenodo.20635335
```

Place it in the project root. This is the fully merged and KeyBERT-annotated dataset (80,814 main-track papers).

If you want to rebuild from raw sources instead, follow the pipeline below.

### Step 2 — (Optional) Rebuild from raw sources

```bash
# Fetch venue-specific papers
python fetch_icml_pmlr_only.py
python fetch_cvpr_2019_2020.py
python fetch_neurips_2017_2020.py
# For ICLR 2017, papers were scraped from OpenReview
# (https://openreview.net/group?id=ICLR.cc/2017/conference)

# Merge into one workbook
python merge_papers.py

# Fill missing abstracts (optional; uses web scraping + Semantic Scholar)
python fill_missing_abstracts.py

# Run KeyBERT keyword extraction (requires full deps; slow on CPU)
python create_keywords_keybert_title.py
```

> Respect each site's terms of use and rate limits when scraping.

### Step 3 — Regenerate figures

```bash
python rebuild_figures.py
```

Outputs go to `topic_analysis_figures_keybert/`. The script reads directly from `merged_papers_keywords_keybert.xlsx` and saves both PNG (300 DPI) and PDF (vector) for each figure.

For all figures including word clouds and supplementary charts:

```bash
python topic_analysis.py
```

### Step 4 — Run the backtest

```bash
python backtest_signature.py
```

Reads `topic_year_venue_counts.csv` (also on Zenodo). Prints the confusion matrix, precision, recall, base rate, named false positives/negatives, and paste-ready LaTeX for Table 2 in the paper.

**Do not change the threshold constants** in `backtest_signature.py` to improve results — those are the pre-committed values stated in the paper. Changing them reintroduces leakage and invalidates the out-of-sample claim.

---

## Citation

If you use this code, dataset, or methodology in published work, please cite:

```bibtex
@misc{khanbayov2026topics,
  doi = {10.5281/ZENODO.20635334},
  url = {https://zenodo.org/doi/10.5281/zenodo.20635334},
  author = {Khanbayov, Rasul and KURBAN, HASAN},
  title = {Topical Phase Transitions in Artificial Intelligence Research: Large-Scale Evidence and an Early-Warning Signature for Emerging Topics},
  publisher = {Zenodo},
  year = {2026}, 
  copyright = {Creative Commons Attribution 4.0 International}
}
```

Dataset (Zenodo):

```bibtex
@misc{khanbayov2026topics,
  author    = {Khanbayov, Rasul and Kurban, Hasan},
  title     = {Topical Phase Transitions in Artificial Intelligence Research: Large-Scale Evidence and an Early-Warning Signature for Emerging Topics},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20635335},
  url       = {https://doi.org/10.5281/zenodo.20635335}
}
```

Please also acknowledge the original data sources: ACL Anthology, CVF Open Access, PMLR, OpenReview, and NeurIPS.cc.

---

## License

Code in this repository is released under the **MIT License** — see [LICENSE](LICENSE) for details.

Data (papers, abstracts, keywords) is derived from publicly available conference proceedings. Usage must comply with the terms of each source.
