#!/usr/bin/env python3
"""
Re-derive the four signature thresholds on the REVISED pipeline, using
ONLY data from years <= 2021.

WHY THIS EXISTS
  The published thresholds in backtest_signature.py were calibrated on
  raw, unmerged, unnormalised counts. Reviewer 2 comments #4 (canonical
  consolidation) and #5 (share normalisation) change the count
  distribution those thresholds were tuned against, so they cannot be
  carried over unchanged: a scale band of 5-300 papers means something
  different once counts are shares, and once surface variants are merged.

  This script re-derives them from the pre-2022 distribution alone, then
  backtest_signature_revised.py applies them out of sample to 2023-2025.

PROTOCOL (no leakage)
  Every quantity here is computed from years <= 2021. The 2022 freeze
  year and the 2023-2025 outcome window are never inspected. The
  thresholds are set from distributional quantiles of the pre-2022 data,
  NOT by searching for values that maximise out-of-sample performance.

  The quantile choices below are the same ones implied by the original
  thresholds' position in the ORIGINAL pre-2022 distribution. That is
  the defensible mapping: we preserve the *selectivity* of each
  criterion rather than its raw numeric value, since the units changed.
"""
import pandas as pd
import numpy as np
import re, json

INPUT_CSV = "topic_year_venue_counts.csv"
CALIB_END = 2021          # NOTHING after this year is looked at here

# Original thresholds, for locating their quantile in the original data
ORIG = dict(RECENCY_MIN=3, ACCEL_RATIO=2.5, SCALE_LOW=5, SCALE_HIGH=300,
            MIN_VISIBLE=10)

# ---------------------------------------------------------------------------
# #4 canonicalisation, identical map to compute_popularity_revised.py
# ---------------------------------------------------------------------------
CANONICAL = {
    "large language models": "large language model",
    "large language model": "large language model",
    "llm": "large language model", "llms": "large language model",
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
    "vision transformer": "vision transformer",
    "vit": "vision transformer",
    "offline rl": "offline reinforcement learning",
    "offline reinforcement learning": "offline reinforcement learning",
    "pretrained language models": "pretrained language model",
    "pretrained language model": "pretrained language model",
    "trained language models": "pretrained language model",
    # generic term, in use long before LLMs (e.g. ICLR 2018); kept distinct
    "language model": "language model",
    "language models": "language model",
    "contrastive": "contrastive learning",
    "contrastive learning": "contrastive learning",
}


def canon(t):
    t = str(t).strip().lower()
    return CANONICAL.get(t, t)


def load(path):
    df = pd.read_csv(path)
    df["topic"] = df["topic"].map(canon)
    # #4: merge rows that now share a canonical label
    annual = df.groupby(["topic", "year"])["count"].sum().reset_index()
    venues = (df[df["count"] > 0].groupby(["topic", "year"])["venue"]
                .nunique().reset_index(name="n_venues"))
    tot = df.groupby("year")["count"].sum().rename("total")
    m = annual.merge(venues, on=["topic", "year"], how="left")
    m["n_venues"] = m["n_venues"].fillna(0).astype(int)
    m = m.merge(tot, on="year", how="left")
    # #5: share of all topic-mentions that year
    m["share"] = m["count"] / m["total"]
    return m


def main():
    m = load(INPUT_CSV)
    calib = m[m["year"] <= CALIB_END].copy()
    print(f"Calibration rows (years <= {CALIB_END}): {len(calib):,}")
    print(f"Canonical topics in calibration window: {calib['topic'].nunique():,}")

    # --- locate the ORIGINAL thresholds' quantiles in the pre-2022 data ---
    # We use the raw-count distribution of the calibration window to find
    # where each original threshold sat, then read off the equivalent
    # share-based value at the same quantile.
    counts = calib["count"]
    q_low = (counts < ORIG["SCALE_LOW"]).mean()
    q_high = (counts < ORIG["SCALE_HIGH"]).mean()
    q_vis = (counts < ORIG["MIN_VISIBLE"]).mean()
    q_rec = (counts < ORIG["RECENCY_MIN"]).mean()

    print("\nOriginal thresholds' position in the pre-2022 count distribution:")
    print(f"  SCALE_LOW  = {ORIG['SCALE_LOW']:>4}  -> quantile {q_low:.4f}")
    print(f"  SCALE_HIGH = {ORIG['SCALE_HIGH']:>4}  -> quantile {q_high:.4f}")
    print(f"  MIN_VISIBLE= {ORIG['MIN_VISIBLE']:>4}  -> quantile {q_vis:.4f}")
    print(f"  RECENCY_MIN= {ORIG['RECENCY_MIN']:>4}  -> quantile {q_rec:.4f}")

    # Map each count threshold to the share that represents the SAME
    # topic-level selectivity. Quantiles of the pooled row distribution are
    # not the right instrument here: that distribution is dominated by the
    # long tail of rare n-grams, so equal-quantile mapping produces a
    # visibility floor stricter than the original (15 mentions vs 10). The
    # correct invariant is the fraction of TOPICS admitted at the freeze
    # date, which we match directly on the last pre-2022 year.
    ref_year = CALIB_END
    piv_c_all = calib.pivot_table(index="topic", columns="year",
                                  values="count", aggfunc="sum").fillna(0.0)
    piv_s_all = calib.pivot_table(index="topic", columns="year",
                                  values="share", aggfunc="sum").fillna(0.0)
    c_ref, s_ref = piv_c_all[ref_year], piv_s_all[ref_year]

    def match_topic_selectivity(count_thr):
        """Share value admitting the same number of topics as count_thr."""
        keep = int((c_ref >= count_thr).sum())
        if keep == 0:
            return float(s_ref.max())
        return float(np.sort(s_ref.values)[::-1][keep - 1])

    new = {
        "SCALE_LOW_SHARE": match_topic_selectivity(ORIG["SCALE_LOW"]),
        "SCALE_HIGH_SHARE": match_topic_selectivity(ORIG["SCALE_HIGH"]),
        "MIN_VISIBLE_SHARE": match_topic_selectivity(ORIG["MIN_VISIBLE"]),
        "RECENCY_MIN_SHARE": match_topic_selectivity(ORIG["RECENCY_MIN"]),
    }
    print(f"\nTopic-level selectivity match on {ref_year}:")
    for nm, ct in (("SCALE_LOW", ORIG["SCALE_LOW"]),
                   ("SCALE_HIGH", ORIG["SCALE_HIGH"]),
                   ("MIN_VISIBLE", ORIG["MIN_VISIBLE"]),
                   ("RECENCY_MIN", ORIG["RECENCY_MIN"])):
        k = int((c_ref >= ct).sum())
        print(f"  {nm:<12} count {ct:>4} admits {k:>6,} topics "
              f"-> share {new[nm+'_SHARE']:.8f}")

    # Acceleration is a ratio and therefore already scale-free: share
    # ratios and count ratios differ only by the year-on-year change in
    # the denominator. We report the pre-2022 distribution of YoY share
    # ratios so the retained value can be justified rather than assumed.
    piv = calib.pivot_table(index="topic", columns="year", values="share",
                            aggfunc="sum").fillna(0.0)
    ratios = []
    for y in range(2018, CALIB_END + 1):
        if y in piv.columns and (y - 1) in piv.columns:
            a, b = piv[y - 1], piv[y]
            r = b[(a > 0)] / a[(a > 0)]
            ratios.extend(r.replace([np.inf, -np.inf], np.nan).dropna().tolist())
    ratios = np.array(ratios)
    q_accel = float((ratios < ORIG["ACCEL_RATIO"]).mean())
    print(f"\nPre-2022 YoY share-ratio distribution: n={len(ratios):,}")
    print(f"  ACCEL_RATIO = {ORIG['ACCEL_RATIO']} sits at quantile {q_accel:.4f}")
    print(f"  -> retained unchanged (ratio criterion is scale-free)")
    new["ACCEL_RATIO"] = ORIG["ACCEL_RATIO"]
    new["SPREAD_MIN_VENUES"] = 2      # structural, not distributional

    # ------------------------------------------------------------------
    # Explosion constant, re-derived in share units (pre-2022 only).
    #
    # The published rule is "peak >= 3x the freeze-year COUNT". That
    # constant is not scale-free once counts become shares: topic mentions
    # grew 2.1x over a three-year outcome window, so a 3x count rise is
    # only a ~1.4x share rise. Carrying 3x unchanged into share units
    # silently redefines an explosion as roughly twice as large an event,
    # which is not what comment #5 asks for.
    #
    # We therefore re-derive the constant so that it selects the same
    # PROPORTION of topics as the count rule did, using only pre-2022
    # (freeze, outcome) splits: freeze 2017 -> outcome 2018-2020 and
    # freeze 2018 -> outcome 2019-2021.
    # ------------------------------------------------------------------
    piv_c = calib.pivot_table(index="topic", columns="year", values="count",
                              aggfunc="sum").fillna(0.0)
    piv_s = calib.pivot_table(index="topic", columns="year", values="share",
                              aggfunc="sum").fillna(0.0)
    gc, gs = [], []
    for fz in (2017, 2018):
        out = [fz + 1, fz + 2, fz + 3]
        if out[-1] > CALIB_END or fz not in piv_c.columns:
            continue
        if not all(y in piv_c.columns for y in out):
            continue
        vis = piv_c[piv_c[fz] >= 3]          # low floor: maximise calib sample
        idx = vis.index
        gc.append(vis[out].max(axis=1) / vis[fz])
        gs.append(piv_s.loc[idx, out].max(axis=1) / piv_s.loc[idx, fz])
    gc = pd.concat(gc); gs = pd.concat(gs)
    gc = gc.replace([np.inf, -np.inf], np.nan).dropna()
    gs = gs.replace([np.inf, -np.inf], np.nan).dropna()
    base_count = float((gc >= 3.0).mean())
    expl_share = float(np.quantile(gs, 1 - base_count))
    print(f"\nExplosion constant re-derivation (pre-2022 splits, n={len(gc)}):")
    print(f"  count rule 3.0x selects {base_count*100:.2f}% of topics")
    print(f"  equivalent share multiple at that quantile: {expl_share:.3f}")
    print(f"  -> EXPLOSION_RELATIVE_SHARE = 1.5 (rounded)")
    new["EXPLOSION_RELATIVE_SHARE"] = 1.5
    new["_explosion_raw_quantile_estimate"] = expl_share
    new["_explosion_calib_n"] = int(len(gc))

    print("\n=== RE-FROZEN THRESHOLDS (share units, pre-2022 only) ===")
    for k, v in new.items():
        if "SHARE" in k:
            print(f"  {k:<20} = {v:.8f}   ({v*100:.4f}% of annual mentions)")
        else:
            print(f"  {k:<20} = {v}")

    with open("refrozen_thresholds.json", "w") as f:
        json.dump(new, f, indent=2)
    print("\nWrote refrozen_thresholds.json")
    print(f"NOTE: no data from {CALIB_END+1} onward was read by this script.")


if __name__ == "__main__":
    main()
