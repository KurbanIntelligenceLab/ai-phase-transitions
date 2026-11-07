#!/usr/bin/env python3
"""
Out-of-sample backtest for the pre-explosion signature -- REVISED pipeline.

Identical protocol to backtest_signature.py, but operating on the pipeline
revised for QSS reviewer 2 comments #4 and #5:

  #4  Surface variants and acronyms are consolidated to canonical labels
      before any counting, so "gnn"/"graph neural networks",
      "generative model"/"generative models", "explainability"/
      "interpretability" etc. are single topics rather than competing ones.
      In the original backtest several of these appeared as separate false
      positives and false negatives simultaneously.

  #5  All criteria operate on each topic's SHARE of annual topic mentions
      rather than on absolute counts, so that venue expansion (2,445 ->
      19,800 papers) does not by itself push a topic across a threshold.

  Thresholds come from refrozen_thresholds.json, produced by
  refreeze_thresholds.py using ONLY years <= 2021. They are read here and
  never adjusted. The freeze year (2022) and outcome window (2023-2025)
  are untouched during calibration, so the out-of-sample claim holds.

  The explosion rule is unchanged and remains RELATIVE (peak in the
  outcome window >= 3x the freeze-year value), so it is already
  scale-free and is applied to shares without modification.
"""
import pandas as pd
import numpy as np
import json

INPUT_CSV = "topic_year_venue_counts.csv"
THRESHOLDS = "refrozen_thresholds.json"

FREEZE_END = 2022
OUTCOME_START = 2023
OUTCOME_END = 2025

RECENCY_WINDOW = 3            # unchanged, structural
SPREAD_MIN_VENUES = 2         # unchanged, structural

T = json.load(open(THRESHOLDS))
# Explosion constant re-derived in share units on pre-2022 data only, so
# that it selects the same proportion of topics as the published 3x count
# rule. See refreeze_thresholds.py for the derivation.
EXPLOSION_RELATIVE = T["EXPLOSION_RELATIVE_SHARE"]
ACCEL_RATIO = T["ACCEL_RATIO"]
SCALE_LOW = T["SCALE_LOW_SHARE"]
SCALE_HIGH = T["SCALE_HIGH_SHARE"]
MIN_VISIBLE = T["MIN_VISIBLE_SHARE"]
RECENCY_MIN = T["RECENCY_MIN_SHARE"]

CANONICAL = {
    "large language models": "large language model",
    "large language model": "large language model",
    "llm": "large language model", "llms": "large language model",
    # generic term, in use long before LLMs (e.g. ICLR 2018); kept distinct
    "language model": "language model",
    "language models": "language model",
    "diffusion models": "diffusion model", "diffusion model": "diffusion model",
    "deep neural networks": "deep learning", "deep learning": "deep learning",
    "graph neural networks": "graph neural network",
    "graph neural network": "graph neural network",
    "gnn": "graph neural network", "gnns": "graph neural network",
    "generative models": "generative model", "generative model": "generative model",
    "interpretability": "interpretability", "explainability": "interpretability",
    "self supervised learning": "self-supervised learning",
    "self-supervised learning": "self-supervised learning",
    "vision language models": "vision language model",
    "vision language model": "vision language model",
    "vision-language models": "vision language model",
    "vision-language model": "vision language model",
    "vlm": "vision language model", "vlms": "vision language model",
    "named entity recognition": "named entity recognition",
    "ner": "named entity recognition",
    "neural machine translation": "neural machine translation",
    "nmt": "neural machine translation",
    "retrieval augmented generation": "retrieval augmented generation",
    "retrieval-augmented generation": "retrieval augmented generation",
    "rag": "retrieval augmented generation",
    "neural radiance fields": "neural radiance field",
    "neural radiance field": "neural radiance field",
    "nerf": "neural radiance field",
    "3d gaussian splatting": "3d gaussian splatting",
    "gaussian splatting": "3d gaussian splatting",
    "vision transformers": "vision transformer",
    "vision transformer": "vision transformer", "vit": "vision transformer",
    "offline rl": "offline reinforcement learning",
    "offline reinforcement learning": "offline reinforcement learning",
    "pretrained language models": "pretrained language model",
    "pretrained language model": "pretrained language model",
    "trained language models": "pretrained language model",
    "contrastive": "contrastive learning",
    "contrastive learning": "contrastive learning",
}


def canon(t):
    return CANONICAL.get(str(t).strip().lower(), str(t).strip().lower())


def load_shares(path):
    df = pd.read_csv(path)
    df["topic"] = df["topic"].map(canon)
    annual = df.groupby(["topic", "year"])["count"].sum().reset_index()
    venues = (df[df["count"] > 0].groupby(["topic", "year"])["venue"]
                .nunique().reset_index(name="n_venues"))
    tot = df.groupby("year")["count"].sum().rename("total")
    m = annual.merge(venues, on=["topic", "year"], how="left")
    m["n_venues"] = m["n_venues"].fillna(0).astype(int)
    m = m.merge(tot, on="year", how="left")
    m["share"] = m["count"] / m["total"]
    return m


def series_for(m, topic):
    sub = m[m["topic"] == topic]
    return {int(r["year"]): (float(r["share"]), int(r["n_venues"]), float(r["count"]))
            for _, r in sub.iterrows()}


def share_at(s, y):
    return s.get(y, (0.0, 0, 0.0))[0]


def venues_at(s, y):
    return s.get(y, (0.0, 0, 0.0))[1]


def signature_positive(s, as_of=FREEZE_END):
    c_now = share_at(s, as_of)
    first = None
    for y in sorted(s):
        if y <= as_of and share_at(s, y) >= RECENCY_MIN:
            first = y
            break
    recency = first is not None and first >= as_of - RECENCY_WINDOW + 1

    def ratio(y0, y1):
        a, b = share_at(s, y0), share_at(s, y1)
        return (b / a) if a > 0 else (np.inf if b > 0 else 0.0)

    accel = (ratio(as_of - 1, as_of) >= ACCEL_RATIO or
             ratio(as_of - 2, as_of - 1) >= ACCEL_RATIO)
    spread = venues_at(s, as_of) >= SPREAD_MIN_VENUES
    scale = SCALE_LOW <= c_now <= SCALE_HIGH
    crit = {"recency": recency, "acceleration": accel,
            "spread": spread, "scale": scale}
    return all(crit.values()), crit


def exploded(s):
    base = share_at(s, FREEZE_END)
    if base == 0:
        return False
    peak = max(share_at(s, y) for y in range(OUTCOME_START, OUTCOME_END + 1))
    return peak >= EXPLOSION_RELATIVE * base


def main():
    m = load_shares(INPUT_CSV)
    fz = m[m["year"] == FREEZE_END].groupby("topic")["share"].sum()
    visible = set(fz[fz >= MIN_VISIBLE].index)
    topics = sorted(t for t in m["topic"].unique() if t in visible)
    print(f"Thresholds loaded from {THRESHOLDS} (frozen on years <= 2021)")
    print(f"Topics with share >= {MIN_VISIBLE:.6f} in {FREEZE_END}: {len(topics)}")

    rows = []
    for t in topics:
        s = series_for(m, t)
        pos, crit = signature_positive(s)
        rows.append({"topic": t, "predicted_positive": pos,
                     "exploded": exploded(s), **crit})
    res = pd.DataFrame(rows)

    TP = int((res.predicted_positive & res.exploded).sum())
    FP = int((res.predicted_positive & ~res.exploded).sum())
    FN = int((~res.predicted_positive & res.exploded).sum())
    TN = int((~res.predicted_positive & ~res.exploded).sum())
    N = TP + FP + FN + TN
    precision = TP / (TP + FP) if (TP + FP) else float("nan")
    recall = TP / (TP + FN) if (TP + FN) else float("nan")
    base_rate = (TP + FN) / N if N else float("nan")

    print("=" * 62)
    print("CONFUSION MATRIX (rows = 2022 prediction, cols = 2023-2025)")
    print(f"  TP={TP}  FP={FP}")
    print(f"  FN={FN}  TN={TN}   (N={N} topics)")
    print("-" * 62)
    print(f"  precision = {precision:.1%}")
    print(f"  recall    = {recall:.1%}")
    print(f"  base rate = {base_rate:.1%}")
    print("-" * 62)
    fp = res[res.predicted_positive & ~res.exploded].topic.tolist()
    fn = res[~res.predicted_positive & res.exploded].topic.tolist()
    print(f"  false positives ({len(fp)}): {fp}")
    print(f"  false negatives ({len(fn)}): {fn}")
    print(f"  true positives  ({TP}): "
          f"{res[res.predicted_positive & res.exploded].topic.tolist()}")
    print("=" * 62)
    print("\n--- paste into Table~\\ref{tab:confusion} ---")
    print(f"Positive (all 4 criteria met) & {TP}~(TP) & {FP}~(FP) \\\\")
    print(f"Negative (criteria not met)   & {FN}~(FN) & {TN}~(TN) \\\\")
    print("\n--- validation sentence ---")
    print(f"precision of {precision*100:.0f}\\% ... recall of {recall*100:.0f}\\% "
          f"... base explosion rate of {base_rate*100:.1f}\\%")

    res.to_csv("backtest_per_topic_revised.csv", index=False)
    print("\nPer-topic detail -> backtest_per_topic_revised.csv")


if __name__ == "__main__":
    main()
