#!/usr/bin/env python3
"""Regenerate Fig. B.12 (top-25 topic labels per venue over years) with
non-topical labels removed. A label is excluded if it names a venue,
event or task identifier rather than a research topic (e.g. "association
computational linguistics", "semeval 2023 task", "2025 task 11"), is a
single generic word ("machine"), or is a KeyBERT fragment beginning with
"based". Each paper counts once per label.
Output: images/09_line_top25_topics_years_<VENUE>.{pdf,png}"""
import os, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "merged_papers_keywords_keybert.xlsx")
CACHE = os.path.join(HERE, "images", "_papers_with_keywords_cache.pkl")
OUT = os.path.join(HERE, "images")
TOP_N = 25
os.makedirs(OUT, exist_ok=True)

GENERIC = {"machine", "task", "tasks", "learning", "model", "models", "data", "method",
           "methods", "approach", "paper", "analysis", "system", "systems", "network",
           "networks", "neural", "language", "deep"}
JUNK_RX = re.compile(r"(\b(19|20)\d\d\b|\btask \d+|semeval|association computational|"
                     r"computational linguisti|meeting association|annual meeting|"
                     r"proceedings|workshop|conference|^based\b)")

def is_junk(k):
    return k in GENERIC or bool(JUNK_RX.search(k))

if os.path.exists(CACHE):
    df = pd.read_pickle(CACHE)
else:
    df = pd.read_excel(XLSX, usecols=["conference", "year", "paper_type", "title", "keywords"])
    df.to_pickle(CACHE)
df = df[df["paper_type"].astype(str).str.lower() == "main"].copy()
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
ex = df.assign(k=df["keywords"].fillna("").astype(str).str.lower().str.split(r";\s*")).explode("k")
ex["k"] = ex["k"].str.strip()
ex = ex[(ex["k"] != "") & ~ex["k"].map(is_junk)]
ex = ex.drop_duplicates(subset=["conference", "year", "title", "k"])

colors = [plt.cm.tab20(i / 19) for i in range(20)] + [plt.cm.tab20b(i / 19) for i in range(20)]
years = list(range(2017, 2026))
for conf in ["ACL", "CVPR", "ICLR", "ICML", "NeurIPS"]:
    sub = ex[ex["conference"] == conf]
    top = sub["k"].value_counts().head(TOP_N).index.tolist()
    mat = (sub[sub["k"].isin(top)].groupby(["k", "year"]).size()
           .unstack("year").reindex(index=top, columns=years).fillna(0).astype(int))
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, kw in enumerate(top):
        ax.plot(years, mat.loc[kw].values, marker="o", markersize=3, label=kw,
                color=colors[i % len(colors)], linewidth=1.5)
    ax.set_xlabel("Year"); ax.set_ylabel("Number of papers")
    ax.set_title(f"Main conference: top {TOP_N} topic labels over years — {conf}")
    ax.set_xticks(years); ax.grid(True, color="#dddddd")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, ncol=1)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUT, f"09_line_top25_topics_years_{conf}.{ext}"), bbox_inches="tight")
    plt.close()
    print(conf, "top labels:", ", ".join(top[:8]), "...")
excluded = df.assign(k=df["keywords"].fillna("").astype(str).str.lower().str.split(r";\s*")).explode("k")
excluded = excluded[excluded["k"].str.strip().map(lambda k: bool(k) and is_junk(k))]
print("\nmost frequent excluded labels:", excluded["k"].str.strip().value_counts().head(10).to_dict())
