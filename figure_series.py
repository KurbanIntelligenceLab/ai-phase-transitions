#!/usr/bin/env python3
"""
Exact data behind the plotted (TikZ) figures of the paper.

Figures 2-11 of the paper are drawn natively in LaTeX/pgfplots, so their
values are typed into main.tex. This script recomputes every one of those
series from the per-paper topic labels, so each plotted number can be
checked against the data.

Counting rules
  * A paper counts once per distinct label it carries (labels are
    lower-cased; duplicates within a paper are dropped).
  * "Mentions" of a phrase (Figs. 5, 9, 10) = number of (paper, label)
    pairs whose label contains the phrase; a paper with two matching
    labels counts twice.
  * Single-label series (Figs. 6, 7, 8, 11) = number of papers carrying
    exactly that label.

Input : merged_papers_keywords_keybert.xlsx   (from the Zenodo deposit)
Output: printed tables; figure_series.json
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "merged_papers_keywords_keybert.xlsx")
YEARS = list(range(2017, 2026))
VENUES = ["ACL", "CVPR", "ICLR", "ICML", "NeurIPS"]

df = pd.read_excel(XLSX, usecols=["conference", "year", "paper_type", "title", "keywords"])
df = df[df["paper_type"].astype(str).str.lower() == "main"].copy()
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
ex = df.assign(k=df["keywords"].fillna("").astype(str).str.lower().str.split(r";\s*")).explode("k")
ex["k"] = ex["k"].astype(str).str.strip()
ex = ex[ex["k"] != ""].drop_duplicates(subset=["conference", "year", "title", "k"])


def by_year(d):
    g = d.groupby("year").size()
    return [int(g.get(y, 0)) for y in YEARS]


def mentions(phrase, venue=None):
    d = ex[ex["k"].str.contains(phrase, regex=False)]
    return by_year(d if venue is None else d[d["conference"] == venue])


def label(lab, venue=None):
    d = ex[ex["k"] == lab]
    return by_year(d if venue is None else d[d["conference"] == venue])


out = {
    "Fig2_papers_per_year_total": by_year(df),
    "Fig2_4_papers_per_year_by_venue": {v: by_year(df[df["conference"] == v]) for v in VENUES},
    "Fig5_mentions_language_model": mentions("language model"),
    "Fig9_mentions_diffusion": mentions("diffusion"),
    "Fig10a_mentions_reinforcement_all": mentions("reinforcement"),
    "Fig10b_mentions_reinforcement_ICLR": mentions("reinforcement", "ICLR"),
    "Fig10c_mentions_reinforcement_NeurIPS": mentions("reinforcement", "NeurIPS"),
    "Fig8_top5_labels": {l: label(l) for l in ["large language models", "reinforcement learning",
                                                "deep learning", "graph neural networks", "diffusion models"]},
}
top15 = ex["k"].value_counts().head(15).index.tolist()
out["Fig7_heatmap_top15_labels"] = {l: label(l) for l in top15}
out["Fig6_ACL_cumulative"] = {l: int(sum(label(l, "ACL"))) for l in
                              ["large language models", "neural machine translation", "named entity recognition"]}
out["Fig11_top10_per_venue_cumulative"] = {
    v: {k: int(n) for k, n in ex[ex["conference"] == v]["k"].value_counts().head(12).items()} for v in VENUES}

print("years:", YEARS)
for key, val in out.items():
    if isinstance(val, dict):
        print(f"\n{key}")
        for k, v in val.items():
            print(f"  {k:<32} {v}")
    else:
        print(f"\n{key}\n  {val}")
with open(os.path.join(HERE, "figure_series.json"), "w") as fh:
    json.dump({"years": YEARS, **out}, fh, indent=1)
print("\nwrote figure_series.json")
print("Note: Fig. 11 omits the non-topical ACL label 'machine'.")
