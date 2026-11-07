"""
Compute popularity scores for keywords from merged_papers_keywords_keybert.xlsx.

REVISED VERSION addressing QSS reviewer 2 comments #4, #5, #6, #7.

Changes from compute_popularity.py:

  #4 Canonical consolidation. Surface variants and acronyms are mapped to a
     single canonical label BEFORE counting (e.g. "large language models",
     "llm", "llms" -> "large language model"). Matching is exact on the
     canonicalised keyword rather than substring containment, which in the
     original double-counted the plural inside the singular.

  #5 Share normalisation. Counts are divided by the total number of accepted
     main-track papers in the same year, so S_NP measures topical market
     share rather than absolute volume inflated by conference growth
     (2,445 papers in 2017 -> 19,800 in 2025).

  #6 Symmetric windows. The trend ratio uses two equal-length 4-year blocks,
     2017-2020 (early) and 2022-2025 (recent), with 2021 excluded as the
     transition year. The original code used 2017-2020 vs 2021-2025 (4 vs 5)
     while the paper stated 2022-2025 vs 2017-2021 (4 vs 5 the other way);
     both were asymmetric and they disagreed with each other.

  #7 Robust scaling. Min-max is replaced with a rank-based (quantile)
     normalisation, which is insensitive to the extreme upper tail that LLM
     topics create and does not compress the remaining topics toward zero.

Run:  python compute_popularity_revised.py
"""
import re, warnings
from pathlib import Path
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

INPUT = Path("merged_papers_keywords_keybert.xlsx")

# ---------------------------------------------------------------------------
# #4 Canonicalisation map: surface variant -> canonical label.
# Applied to every extracted keyword before counting.
# ---------------------------------------------------------------------------
CANONICAL = {
    # large language models
    "large language models": "large language model",
    "large language model": "large language model",
    "llm": "large language model",
    "llms": "large language model",
    # diffusion
    "diffusion models": "diffusion model",
    "diffusion model": "diffusion model",
    # reinforcement learning
    "reinforcement learning": "reinforcement learning",
    # deep learning
    "deep learning": "deep learning",
    "deep neural networks": "deep learning",
    # graph neural networks
    "graph neural networks": "graph neural network",
    "graph neural network": "graph neural network",
    "gnn": "graph neural network",
    "gnns": "graph neural network",
    # representation learning
    "representation learning": "representation learning",
    # generative models
    "generative models": "generative model",
    "generative model": "generative model",
    # interpretability
    "interpretability": "interpretability",
    "explainability": "interpretability",
    # self-supervised learning
    "self-supervised learning": "self-supervised learning",
    "self supervised learning": "self-supervised learning",
    # vision language models
    "vision language models": "vision language model",
    "vision language model": "vision language model",
    "vision-language models": "vision language model",
    "vision-language model": "vision language model",
    "vlm": "vision language model",
    "vlms": "vision language model",
    # named entity recognition
    "named entity recognition": "named entity recognition",
    "ner": "named entity recognition",
    # neural machine translation
    "neural machine translation": "neural machine translation",
    "nmt": "neural machine translation",
    # 3d object detection
    "3d object detection": "3d object detection",
    # federated learning
    "federated learning": "federated learning",
    # robustness
    "robustness": "robustness",
    # retrieval augmented generation
    "retrieval augmented generation": "retrieval augmented generation",
    "retrieval-augmented generation": "retrieval augmented generation",
    "rag": "retrieval augmented generation",
    # 3d gaussian splatting
    "3d gaussian splatting": "3d gaussian splatting",
    "gaussian splatting": "3d gaussian splatting",
    # neural radiance fields
    "neural radiance fields": "neural radiance field",
    "neural radiance field": "neural radiance field",
    "nerf": "neural radiance field",
}

# Canonical topics to score (deduplicated targets of the map above)
TOPICS = sorted(set(CANONICAL.values()))

# ---------------------------------------------------------------------------
# #6 Symmetric 4-year windows; 2021 excluded as the transition year.
# ---------------------------------------------------------------------------
EARLY_YEARS = [2017, 2018, 2019, 2020]
RECENT_YEARS = [2022, 2023, 2024, 2025]

print("Loading...")
df = pd.read_excel(INPUT, engine="openpyxl")
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
main = df[df["paper_type"].astype(str).str.strip().str.lower() == "main"].copy()
main = main[main["keywords"].notna()].copy()
main = main[~main["keywords"].astype(str).str.strip().str.startswith("Topic:")].copy()


def parse_kw(s):
    """Split the keywords cell and canonicalise each surface form (#4)."""
    if pd.isna(s):
        return []
    out = []
    for k in re.split(r"[;,]", str(s)):
        k = k.strip().lower()
        if not k:
            continue
        out.append(CANONICAL.get(k, k))
    return out


main["_kw"] = main["keywords"].apply(parse_kw)

years = sorted([y for y in main["year"].dropna().unique() if 2017 <= y <= 2025])

# #5 Denominator: total accepted main-track papers per year.
papers_per_year = {int(y): int((main["year"] == y).sum()) for y in years}
print("\nPapers per year (normalisation denominator, #5):")
for y in years:
    print(f"  {int(y)}: {papers_per_year[int(y)]:,}")


def count_topic_year(topic):
    """Exact match on canonical label (#4), not substring containment."""
    counts = {}
    for y in years:
        sub = main[main["year"] == y]
        n = sum(1 for kws in sub["_kw"] for k in kws if k == topic)
        counts[int(y)] = n
    return counts


def rank_normalise(s):
    """#7 Robust rank-based scaling to [0,1]; ties averaged."""
    if len(s) <= 1:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s.rank(method="average") - 1) / (len(s) - 1)


print("\nComputing scores for all topics...")
records = []
for topic in TOPICS:
    cy = count_topic_year(topic)

    # #5 share of all papers in that year, averaged over the window
    shares = [cy.get(y, 0) / papers_per_year[y] for y in years if papers_per_year[y]]
    avg_share = float(np.mean(shares)) if shares else 0.0

    # #6 symmetric 4-year windows, computed on shares so that conference
    # growth does not inflate the ratio
    early_share = sum(cy.get(y, 0) / papers_per_year[y]
                      for y in EARLY_YEARS if papers_per_year.get(y))
    recent_share = sum(cy.get(y, 0) / papers_per_year[y]
                       for y in RECENT_YEARS if papers_per_year.get(y))
    # A topic absent from the early window has an undefined ratio. Rather
    # than dropping it or assigning infinity (which collapses several
    # distinct topics onto one tied value), apply additive smoothing: one
    # paper's worth of share in the earliest year of the early window. This
    # keeps the ordering among such topics driven by their actual recent
    # share instead of by an arbitrary shared cap.
    if early_share > 0:
        trend = recent_share / early_share
        smoothed = False
    else:
        floor = 1.0 / papers_per_year[EARLY_YEARS[0]]
        trend = recent_share / floor
        smoothed = True

    records.append({
        "topic": topic,
        "avg_share": avg_share,
        "trend": trend,
        "count_2025": cy.get(2025, 0),
        "share_2025": cy.get(2025, 0) / papers_per_year[2025] if papers_per_year.get(2025) else 0.0,
        "early_zero": smoothed,
    })

df_r = pd.DataFrame(records)

# #7 robust rank-based normalisation in place of min-max
df_r["S_NP"] = rank_normalise(df_r["avg_share"])
df_r["S_NTr"] = rank_normalise(df_r["trend"])
df_r["P_k"] = df_r["S_NP"] + df_r["S_NTr"]

df_r = df_r.sort_values("P_k", ascending=False).reset_index(drop=True)
df_r["rank"] = df_r.index + 1

print("\n=== POPULARITY TABLE (revised: #4 #5 #6 #7) ===")
print(f"{'Rank':<5} {'Topic':<32} {'S_NP':>6} {'S_NTr':>6} {'P_k':>6} "
      f"{'2025 n':>8} {'2025 %':>8}")
print("-" * 80)
for _, r in df_r.iterrows():
    flag = " *" if r["early_zero"] else ""
    print(f"{int(r['rank']):<5} {r['topic']:<32} {r['S_NP']:>6.2f} {r['S_NTr']:>6.2f} "
          f"{r['P_k']:>6.2f} {int(r['count_2025']):>8,} {100*r['share_2025']:>7.2f}%{flag}")
print("\n* topic absent in the 2017-2020 window; trend ratio computed against "
      "an additive-smoothing floor of one paper in 2017.")

df_r.to_csv("popularity_revised.csv", index=False)
print("\nWrote popularity_revised.csv")
