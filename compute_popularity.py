"""
Compute popularity scores for keywords from merged_papers_keywords_keybert.xlsx.
Matches the paper's methodology:
  S_NP  = normalised average proportion across all years
  S_NTr = normalised trend score: sum(2021-2025) / sum(2017-2020)
  P_k   = S_NP + S_NTr
"""
import re, warnings
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

INPUT = Path("merged_papers_keywords_keybert.xlsx")

# Topics to score (normalised lowercase for matching)
TOPICS = [
    "large language models",
    "reinforcement learning",
    "deep learning",
    "diffusion models",
    "graph neural networks",
    "representation learning",
    "large language model",
    "diffusion model",
    "generative models",
    "interpretability",
    "self-supervised learning",
    "vision language models",
    "named entity recognition",
    "neural machine translation",
    "3d object detection",
    "federated learning",
    "robustness",
    "retrieval augmented generation",
    "3d gaussian splatting",
    "neural radiance fields",
]

print("Loading...")
df = pd.read_excel(INPUT, engine="openpyxl")
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
main = df[df["paper_type"].astype(str).str.strip().str.lower() == "main"].copy()
main = main[main["keywords"].notna()].copy()
main = main[~main["keywords"].astype(str).str.strip().str.startswith("Topic:")].copy()

def parse_kw(s):
    if pd.isna(s): return []
    return [k.strip().lower() for k in re.split(r"[;,]", str(s)) if k.strip()]

main["_kw"] = main["keywords"].apply(parse_kw)

years = sorted([y for y in main["year"].dropna().unique() if 2017 <= y <= 2025])
early = [y for y in years if y <= 2020]
late  = [y for y in years if y >= 2021]

# Count per topic per year
def count_topic_year(topic):
    counts = {}
    for y in years:
        sub = main[main["year"] == y]
        n = sum(1 for kws in sub["_kw"] for k in kws if topic in k)
        counts[y] = n
    return counts

print("Computing scores for all topics...")
records = []
for topic in TOPICS:
    cy = count_topic_year(topic)
    avg_prop = np.mean(list(cy.values()))  # average count (proxy for proportion)
    early_sum = sum(cy.get(y, 0) for y in early)
    late_sum  = sum(cy.get(y, 0) for y in late)
    trend = late_sum / early_sum if early_sum > 0 else late_sum + 1
    count_2025 = cy.get(2025, 0)
    records.append({
        "topic": topic,
        "avg_prop": avg_prop,
        "trend": trend,
        "count_2025": count_2025,
    })

df_r = pd.DataFrame(records)

# Normalise
df_r["S_NP"]  = (df_r["avg_prop"] - df_r["avg_prop"].min()) / (df_r["avg_prop"].max() - df_r["avg_prop"].min())
df_r["S_NTr"] = (df_r["trend"]    - df_r["trend"].min())    / (df_r["trend"].max()    - df_r["trend"].min())
df_r["P_k"]   = df_r["S_NP"] + df_r["S_NTr"]

# Sort by original topic order (keep user ranking) but show scores
df_r["rank"] = df_r["topic"].apply(lambda t: TOPICS.index(t) + 1)
df_r = df_r.sort_values("rank")

print("\n=== POPULARITY TABLE ===")
print(f"{'Rank':<4} {'Topic':<35} {'S_NP':>6} {'S_NTr':>6} {'P_k':>6} {'2025':>7}")
print("-" * 70)
for _, row in df_r.iterrows():
    print(f"{int(row['rank']):<4} {row['topic']:<35} {row['S_NP']:>6.2f} {row['S_NTr']:>6.2f} {row['P_k']:>6.2f} {int(row['count_2025']):>7,}")
