# Topical Phase Transitions in AI Research
### Large-Scale Evidence and an Early-Warning Signature for Emerging Topics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20468041.svg)](https://doi.org/10.5281/zenodo.20468041)

> **Paper submitted to:** *Quantitative Science Studies (QSS)*  
> **Title:** "Topical Phase Transitions in Artificial Intelligence Research: Large-Scale Evidence and an Early-Warning Signature for Emerging Topics"

---

## Overview

This repository contains all code, scripts, and figures supporting the paper. The project tracks **84,091 accepted papers** from five major AI/ML conferences (ACL, CVPR, ICLR, ICML, NeurIPS) over 2017–2025, extracts research topics via KeyBERT, and formalises and validates a **pre-explosion signature** that forecasts which topics will undergo a rapid phase transition (3× growth) within three years.

### Key findings

- AI research volume grew at **30.4% CAGR** (2,445 → 20,459 papers, 2017–2025).
- Large Language Models (LLMs) and diffusion models underwent the most dramatic topical phase transitions post-2022.
- A four-criterion early-warning signature achieves **63% recall** at **27% precision** (vs. 13.5% base rate) in a strict out-of-sample backtest on 223 topics.
- The signature is evaluated at a fixed freeze date (2022) with no peeking at outcome years (2023–2025).

---

## Repository structure

```
paper/
│
├── main.tex                          # Paper source (LaTeX, elsarticle format)
├── paper_draft.tex                   # Archived earlier draft
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
├── merge_and_lda.py                  # Merge + LDA topic modelling (early pipeline)
├── create_keywords_from_titles.py    # Simple bag-of-words keywords from titles
├── create_keywords_vocabulary_match.py # Controlled-vocabulary keyword matching
├── create_keywords_keybert_title.py  # KeyBERT keyword extraction from titles
├── run_keybert_on_lda.py             # Run KeyBERT on LDA-assigned text chunks
│
├── ── Analysis & figures ──
├── topic_analysis.py                 # Master figure script: counts, heatmaps,
│                                     #   LLM/diffusion/RL trends, word clouds
├── create_paper_figures.py           # Additional paper-specific figures
├── rebuild_figures.py                # Regenerate all figures from exact data values
│                                     #   (Figs 05, 06, 07, 12, 14, 15 → v5 folder)
├── analyze_keywords.py               # Keyword frequency and distribution summaries
├── compute_popularity.py             # Popularity scoring (Xiong et al. 2019 method)
├── sample_check.py                   # Spot-check random rows in the dataset
├── inspect_excels.py                 # Quick inspection of Excel file schemas
├── verify_output.py                  # Verify figure output completeness
├── regen_figB.py                     # Regenerate figure B (architecture diagram)
│
├── ── Validation ──
├── backtest_signature.py             # Out-of-sample backtest of the pre-explosion
│                                     #   signature (Section 5 of the paper)
├── check_conflicts.py                # Trace and resolve data/figure conflicts
├── check_qss.py                      # QSS submission compliance checker
│
├── ── Environment ──
├── environment.yml                   # Conda environment specification
├── requirements.txt                  # pip dependencies (core)
├── requirements-full.txt             # pip dependencies (full: KeyBERT + PyTorch)
│
├── ── Figures (final, committed) ──
├── topic_analysis_figures_keybert_v5/  # All figures used in the paper (PDF + PNG)
│   ├── 05_top10_topics_per_conference.{png,pdf}
│   ├── 06_heatmap_topics_over_years.{png,pdf}
│   ├── 07_line_top5_topics_over_years.{png,pdf}
│   ├── 10_exponential_growth_total_and_stacked.{png,pdf}
│   ├── 11_top25_per_venue_*.{png,pdf}     (one per venue)
│   ├── 12_llm_revolution_over_years.{png,pdf}
│   ├── 13_diffusion_rise_over_years.{png,pdf}
│   ├── 14_diffusion_rise_over_years.{png,pdf}
│   ├── 15_rl_over_years_all.{png,pdf}
│   └── verified_topic_counts_by_year.csv  (audit trail)
│
├── paper_figures/                    # Additional figures (A–D): word clouds,
│                                     #   mindmap, keyphrases, topic distribution
│
├── architecture.png                  # System architecture diagram (Fig. in paper)
│
└── PROMPT_conference_review_paper.md # Prompt template for AI-assisted drafting
```

> **Data files** (`.xlsx`, `.csv`) are excluded from Git via `.gitignore` due to size (total ~500 MB). The canonical dataset is archived at Zenodo: [doi:10.5281/zenodo.20468041](https://doi.org/10.5281/zenodo.20468041).

---

## Dataset

| Attribute | Value |
|---|---|
| Total papers | **84,091** |
| Venues | ACL, CVPR, ICLR, ICML, NeurIPS |
| Years | 2017–2025 |
| Keyword method | KeyBERT (`all-MiniLM-L6-v2`), MMR diversity=0.7, n-gram (1,3), top-12 terms |
| Source | Official proceedings (ACL Anthology, CVF, PMLR, OpenReview, NeurIPS.cc) |
| Zenodo archive | [doi:10.5281/zenodo.20468041](https://doi.org/10.5281/zenodo.20468041) |

**Venue breakdown (2025):**

| Venue | Papers (2025) | Share |
|---|---|---|
| NeurIPS | 5,486 | 26.8% |
| CVPR | 4,730 | 23.1% |
| ACL | 4,297 | 21.0% |
| ICML | 3,215 | 15.7% |
| ICLR | 2,731 | 13.3% |

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
https://doi.org/10.5281/zenodo.20468041
```

Place it in the project root (`paper/`). This is the fully merged and KeyBERT-annotated dataset (84,091 rows).

If you want to rebuild from raw sources instead, follow the pipeline below.

### Step 2 — (Optional) Rebuild from raw sources

```bash
# Fetch venue-specific papers
python fetch_icml_pmlr_only.py
python fetch_cvpr_2019_2020.py
python fetch_neurips_2017_2020.py
# For ICLR 2017 specifically, papers were scraped from OpenReview
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

Outputs go to `topic_analysis_figures_keybert_v5/`. The script reads directly from `merged_papers_keywords_keybert.xlsx` and saves both PNG (300 DPI) and PDF (vector) for each figure.

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

### Step 5 — Compile the paper

The paper uses the `elsarticle` LaTeX class (QSS house style):

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use your preferred LaTeX IDE (Overleaf, TeXstudio, VS Code + LaTeX Workshop).

---

## Figures in the paper

| Figure | File | Description |
|---|---|---|
| Fig. 1 | `paper_figures/A_*.png` | Word cloud grid of top topics per venue |
| Fig. 2 | `paper_figures/B_*.png` | Research subfield mindmap |
| Fig. 3 | `paper_figures/C_*.png` | Topic distribution overview |
| Fig. 4 | `paper_figures/D_*.png` | Keyphrases summary |
| Fig. 5 | `topic_analysis_figures_keybert_v5/05_top10_topics_per_conference.pdf` | Top-10 topics per venue (horizontal bar) |
| Fig. 6 | TikZ in `main.tex` | ACL: LLM vs NMT vs NER counts |
| Fig. 7 | `topic_analysis_figures_keybert_v5/07_line_top5_topics_over_years.pdf` | Top-5 topics over time (all venues) |
| Fig. 8 | `architecture.png` | System architecture / pipeline |
| Fig. 9 | TikZ in `main.tex` | Popularity scoring diagram |
| Fig. 10 | `topic_analysis_figures_keybert_v5/10_exponential_growth_total_and_stacked.pdf` | Total papers per year (stacked by venue) |
| Fig. 11 | `topic_analysis_figures_keybert_v5/11_top25_per_venue_*.pdf` | Top-25 topics per venue |
| Fig. 12 | `topic_analysis_figures_keybert_v5/12_llm_revolution_over_years.pdf` | LLM-related topics over years |
| Fig. 13 | TikZ in `main.tex` | Pre-explosion signature schematic |
| Fig. 14 | `topic_analysis_figures_keybert_v5/14_diffusion_rise_over_years.pdf` | Diffusion models over years |
| Fig. 15 | `topic_analysis_figures_keybert_v5/15_rl_over_years_all.pdf` | Reinforcement learning over years |

---

## QSS submission compliance

The paper satisfies all Quantitative Science Studies requirements:

| Requirement | Status |
|---|---|
| Abstract ≤ 200 words | 129 words |
| Keywords ≤ 6 | 6 |
| Line numbering enabled | Yes (`\linenumbers`) |
| Data availability + Zenodo DOI | Yes |
| CRediT author contributions | Yes |
| Competing interests statement | Yes |
| Author-year citation style | Yes (`\citep`, `\citet`) |
| Body word count 5,000–10,000 | ~7,534 |
| Methods before Results | Yes |
| No colored table cells | Yes |

---

## Citation

If you use this code, dataset, or methodology in published work, please cite:

```bibtex
@article{bayov2025topical,
  title   = {Topical Phase Transitions in Artificial Intelligence Research:
             Large-Scale Evidence and an Early-Warning Signature for Emerging Topics},
  author  = {Bayov, Rasul and others},
  journal = {Quantitative Science Studies},
  year    = {2025},
  note    = {Submitted}
}
```

Dataset (Zenodo):

```bibtex
@misc{bayov2025dataset,
  author    = {Bayov, Rasul and others},
  title     = {AI Conference Papers Dataset (ACL, CVPR, ICLR, ICML, NeurIPS) 2017--2025},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20468041},
  url       = {https://doi.org/10.5281/zenodo.20468041}
}
```

Please also acknowledge the original data sources: ACL Anthology, CVF Open Access, PMLR, OpenReview, and NeurIPS.cc.

---

## License

Code in this repository is released under the **MIT License** — see [LICENSE](LICENSE) for details.

Data (papers, abstracts, keywords) is derived from publicly available conference proceedings. Usage must comply with the terms of each source.
