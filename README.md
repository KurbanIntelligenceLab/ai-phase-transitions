# Topical Phase Transitions in AI Research
### Large-Scale Evidence and an Early-Warning Signature for Emerging Topics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20635334.svg)](https://doi.org/10.5281/zenodo.20635334)
[![arXiv](https://img.shields.io/badge/arXiv-2606.12828-b31b1b.svg)](https://arxiv.org/abs/2606.12828)

Code for the paper *"Topical Phase Transitions in Artificial Intelligence Research: Large-Scale
Evidence and an Early-Warning Signature for Emerging Topics"* by Rasul Khanbayov and Hasan Kurban,
accepted for publication in **Quantitative Science Studies**.

The data needed to reproduce every figure and table are archived on Zenodo:
**[doi:10.5281/zenodo.20635334](https://doi.org/10.5281/zenodo.20635334)** (this DOI always
resolves to the latest version).

---

## Overview

The project tracks **80,814 accepted main-track papers** from ACL, CVPR, ICLR, ICML and NeurIPS
(2017–2025). Each paper carries a set of topic labels, and topic counts are aggregated by year and
venue to study how AI research reorganizes. The paper's two contributions:

1. **Topical phase transitions.** Major topics such as large language models and diffusion models
   stay marginal for years and then surge across venues within one to three years, whereas
   reinforcement learning compounds smoothly.
2. **An early-warning signature.** Four publication-dynamics criteria (recency, acceleration,
   cross-venue spread, pre-saturation scale), with thresholds frozen on 2017–2021 data, are
   evaluated on held-out 2023–2025 outcomes and then applied to 2025 data.

### Results reported in the paper

| | Value |
|---|---|
| Papers | 80,814 (2,445 in 2017 → 19,800 in 2025) |
| Held-out evaluation (freeze 2022, outcomes 2023–2025) | 189 topics; TP 9 / FP 31 / FN 14 / TN 135 |
| Precision / recall / base rate | **22.5% / 39% / 12.2%** (lift 1.85×) |
| Recall 95% Wilson interval | ≈ 22%–59% (9 of 23 events) |
| Sensitivity (explosion multiplier 1.2×–3.0×) | recall 36%–50%, lift 1.72–2.36 |
| Candidate topics flagged from 2025 data | 46 labels (e.g. LLM reasoning, LLM agents, multimodal LLMs, RAG, flow matching) |

**Important caveat (paper Section 5.3).** Topic labels come from two sources: author-supplied
keywords where a venue publishes them (ICLR in all years, NeurIPS from 2021, ICML in 2025) and
KeyBERT keyphrases of the title elsewhere. Because NeurIPS and ICML change source inside the
window, some topics appear to emerge without any change in research activity. Measured
consistently from titles, only 3 of the 9 true positives remain (LLM, diffusion model, NeRF),
so the signature's out-of-sample performance should not be regarded as established.
`robustness_keyword_source.py` reproduces this check.

---

## Repository structure

```
├── Data collection
│   ├── fetch_icml_pmlr_only.py        Scrape ICML papers from PMLR
│   ├── fetch_icml_papers.py           Fetch ICML via the OpenReview API (alternative)
│   ├── fetch_cvpr_2019_2020.py        Scrape CVPR 2019–2020 listings
│   ├── fetch_neurips_2017_2020.py     Scrape NeurIPS 2017–2020 listings
│   ├── fill_missing_abstracts.py      Fill missing abstracts (used only for extraction)
│   └── merge_papers.py                Merge per-venue files into one workbook
│
├── Topic labels
│   ├── create_keywords_keybert_title.py     KeyBERT keyphrases (all-MiniLM-L6-v2, MMR 0.7, n-grams 1–3)
│   ├── create_keywords_from_titles.py       Simple title keywords (exploratory)
│   └── create_keywords_vocabulary_match.py  Controlled-vocabulary matching (exploratory)
│
├── Analysis used in the paper
│   ├── compute_popularity_revised.py  Popularity ranking, Table A.1  -> popularity_revised.csv
│   ├── refreeze_thresholds.py         Freeze signature thresholds on 2017–2021 data -> refrozen_thresholds.json
│   ├── backtest_signature_revised.py  Held-out evaluation, Table 2   -> backtest_per_topic_revised.csv
│   ├── sensitivity_analysis.py        Threshold sweeps, Appendix C (Table C.1)
│   ├── robustness_keyword_source.py   Keyword-source robustness, Section 5.3 (Table 3)
│   ├── figure_series.py               Exact data behind the plotted Figures 2–11 -> figure_series.json
│   ├── make_figure_b12.py             Figure B.1 (per-venue top-25 labels)
│   └── refrozen_thresholds.json       The frozen thresholds used in the paper
│
├── Exploratory figures (not the published versions)
│   ├── topic_analysis.py, rebuild_figures.py
│   └── topic_analysis_figures_keybert/
│
├── legacy/                            Scripts behind the first-submission numbers (superseded)
└── environment.yml, requirements*.txt
```

---

## Data (Zenodo)

Download these files from [doi:10.5281/zenodo.20635334](https://doi.org/10.5281/zenodo.20635334)
and place them in the repository root:

| File | Contents | Used by |
|---|---|---|
| `merged_papers_keywords_keybert.xlsx` | One row per paper: venue, year, title, topic labels (abstracts not included) | popularity, figures, robustness |
| `topic_year_venue_counts.csv` | Topic-label mentions per topic, year and venue | signature: refreeze, backtest, sensitivity |

**Note on the two files.** `topic_year_venue_counts.csv` was produced by an earlier label-extraction
pass and is not an exact aggregation of the per-paper labels; the two differ mainly for CVPR
2021–2025 and for ACL. The signature results in Section 5 (Tables 2–4, Appendix C) are computed
from the counts file; the descriptive results in Section 4, Figures 5–11, Table A.1 and Figure B.1
are computed from the per-paper labels. Both files are needed to reproduce the paper exactly.

Paper abstracts and full texts are not redistributed. They can be obtained from the original
sources (ACL Anthology, CVF Open Access, PMLR, OpenReview, NeurIPS proceedings) under their terms
of use.

---

## Reproducing the paper

```bash
conda env create -f environment.yml && conda activate ml-conference-papers
# or: pip install -r requirements.txt

python refreeze_thresholds.py          # thresholds from 2017–2021 data only
python backtest_signature_revised.py   # Table 2: 9/31/14/135, precision 22.5%, recall 39%
python sensitivity_analysis.py         # Appendix C
python robustness_keyword_source.py    # Section 5.3 / Table 3
python compute_popularity_revised.py   # Table A.1
python figure_series.py                # values plotted in Figures 2–11
python make_figure_b12.py              # Figure B.1
```

`refreeze_thresholds.py` reads only years ≤ 2021 and reproduces `refrozen_thresholds.json`
exactly. **Do not tune the thresholds** to improve the held-out results; doing so reintroduces
leakage and invalidates the out-of-sample evaluation.

Figures 2–11 are drawn natively in LaTeX (pgfplots); `figure_series.py` prints every plotted value.
Figures 5, 9 and 10 plot label *mentions* (a paper with two matching labels counts twice), as
stated in their captions.

### Rebuilding the dataset from raw sources (optional)

```bash
python fetch_icml_pmlr_only.py
python fetch_cvpr_2019_2020.py
python fetch_neurips_2017_2020.py      # ICLR 2017 was scraped from OpenReview
python merge_papers.py
python fill_missing_abstracts.py       # optional
python create_keywords_keybert_title.py   # requirements-full.txt (KeyBERT + PyTorch)
```

Respect each site's terms of use and rate limits when scraping.

---

## Citation

Please cite the article (Quantitative Science Studies, 2026; DOI to be added on publication) and
the data:

```bibtex
@misc{khanbayov2026topics_data,
  author    = {Khanbayov, Rasul and Kurban, Hasan},
  title     = {Topical Phase Transitions in Artificial Intelligence Research:
               Large-Scale Evidence and an Early-Warning Signature for Emerging Topics},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20635334},
  url       = {https://doi.org/10.5281/zenodo.20635334}
}
```

See also [CITATION.cff](CITATION.cff). Please acknowledge the original data sources: ACL
Anthology, CVF Open Access, PMLR, OpenReview and NeurIPS proceedings.

---

## License

Code: **MIT License** (see [LICENSE](LICENSE)). Data on Zenodo: **CC BY 4.0**. Paper metadata is
derived from publicly available proceedings; use must comply with each source's terms.
