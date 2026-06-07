#!/usr/bin/env python3
"""
Out-of-sample backtest for the pre-explosion signature.

WHAT THIS DOES
  Implements EXACTLY the protocol described in Section 5 of the paper:
    1. Freeze the four signature thresholds using 2017-2021 data only.
    2. Evaluate the signature for every topic AS OF 2021 (no peeking at
       2022-2025).
    3. Use 2022-2025 data and the pre-committed explosion rule to label
       each topic exploded / did-not-explode.
    4. Build the 2x2 confusion matrix; compute precision, recall, base
       rate, and name the false positives.

  The design guarantees NO LEAKAGE: signature evaluation uses only
  years <= 2021; outcome labels use only years >= 2022. The explosion
  rule is fixed in code BEFORE any topic is scored and is never tuned.

INPUT
  A CSV with one row per (topic, year, venue) and a count, OR per-paper
  rows that this script will aggregate. Set INPUT_CSV and the column
  names below. Expected tidy format:
      topic,year,venue,count
  If your data is one row per paper (topic,year,venue with no count),
  set ONE_ROW_PER_PAPER = True and it will group and count.

OUTPUT
  Prints the confusion matrix, precision, recall, base rate, the list
  of false positives and false negatives, and a ready-to-paste LaTeX
  snippet for Table~\ref{tab:confusion} and the surrounding sentence.

IMPORTANT
  The thresholds below are the ones STATED in the paper. They are
  asserted to have been chosen on 2017-2021 data. Do not change them to
  improve the result; that would reintroduce leakage and invalidate the
  out-of-sample claim. If you change a threshold, you must re-justify it
  on 2017-2021 data only and update the paper text to match.
"""

import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# CONFIG  -- edit these to match your data file
# ----------------------------------------------------------------------
INPUT_CSV = "topic_year_venue_counts.csv"
ONE_ROW_PER_PAPER = False          # True if each row is a single paper
COL_TOPIC = "topic"
COL_YEAR = "year"
COL_VENUE = "venue"
COL_COUNT = "count"                # ignored if ONE_ROW_PER_PAPER

FREEZE_END = 2022                  # signature evaluated as of this year
OUTCOME_START = 2023               # explosion outcomes from this year on
OUTCOME_END = 2025

# ---- Pre-committed thresholds (STATED in the paper; do not tune) ----
EXPLOSION_RELATIVE = 3.0           # peak in outcome window >= 3x freeze-year count

RECENCY_MIN = 3                    # >=3 papers/yr first reached...
RECENCY_WINDOW = 3                 # ...within preceding 3 years
ACCEL_RATIO = 2.5                  # >=2.5x YoY in one of last 2 intervals
SPREAD_MIN_VENUES = 2              # appeared in >=2 of 5 venues
SCALE_LOW, SCALE_HIGH = 5, 300     # pre-saturation annual-count band
MIN_VISIBLE = 10                   # min papers at freeze date to enter evaluation


# ----------------------------------------------------------------------
# LOAD AND SHAPE
# ----------------------------------------------------------------------
def load_counts(path):
    df = pd.read_csv(path)
    if ONE_ROW_PER_PAPER:
        df = (df.groupby([COL_TOPIC, COL_YEAR, COL_VENUE])
                .size().reset_index(name=COL_COUNT))
    # annual cross-venue count per topic-year
    annual = (df.groupby([COL_TOPIC, COL_YEAR])[COL_COUNT]
                .sum().reset_index(name="count"))
    # venue presence per topic-year (how many venues had >=1 paper)
    venues = (df[df[COL_COUNT] > 0]
                .groupby([COL_TOPIC, COL_YEAR])[COL_VENUE]
                .nunique().reset_index(name="n_venues"))
    m = annual.merge(venues, on=[COL_TOPIC, COL_YEAR], how="left")
    m["n_venues"] = m["n_venues"].fillna(0).astype(int)
    return m


def series_for(topic_df, topic):
    """Return dict year->(count, n_venues) for one topic, all years."""
    sub = topic_df[topic_df[COL_TOPIC] == topic]
    by_year = {int(r[COL_YEAR]): (float(r["count"]), int(r["n_venues"]))
               for _, r in sub.iterrows()}
    return by_year


def count_at(series, year):
    return series.get(year, (0.0, 0))[0]


def venues_at(series, year):
    return series.get(year, (0.0, 0))[1]


# ----------------------------------------------------------------------
# SIGNATURE  (evaluated using ONLY years <= FREEZE_END)
# ----------------------------------------------------------------------
def signature_positive(series, as_of=FREEZE_END):
    """Return (is_positive, criteria_dict) using only data up to as_of."""
    c_now = count_at(series, as_of)

    # 1. Recency: first reached >= RECENCY_MIN within preceding window
    first_year_at_scale = None
    for y in sorted(series):
        if y <= as_of and count_at(series, y) >= RECENCY_MIN:
            first_year_at_scale = y
            break
    recency = (first_year_at_scale is not None and
               first_year_at_scale >= as_of - RECENCY_WINDOW + 1)

    # 2. Acceleration: >= ACCEL_RATIO YoY in one of the last 2 intervals
    def ratio(y0, y1):
        a, b = count_at(series, y0), count_at(series, y1)
        return (b / a) if a > 0 else (np.inf if b > 0 else 0.0)
    accel = (ratio(as_of - 1, as_of) >= ACCEL_RATIO or
             ratio(as_of - 2, as_of - 1) >= ACCEL_RATIO)

    # 3. Cross-venue spread (as of as_of)
    spread = venues_at(series, as_of) >= SPREAD_MIN_VENUES

    # 4. Pre-saturation scale band (as of as_of)
    scale = SCALE_LOW <= c_now <= SCALE_HIGH

    crit = {"recency": recency, "acceleration": accel,
            "spread": spread, "scale": scale}
    return all(crit.values()), crit


# ----------------------------------------------------------------------
# EXPLOSION OUTCOME  (uses ONLY years >= OUTCOME_START)
# ----------------------------------------------------------------------
def exploded(series):
    """Relative explosion rule: peak count in [OUTCOME_START, OUTCOME_END]
    is at least EXPLOSION_RELATIVE x the freeze-year count.
    A zero baseline means the topic did not yet exist and cannot explode."""
    baseline = count_at(series, FREEZE_END)
    if baseline == 0:
        return False
    peak = max(count_at(series, Y) for Y in range(OUTCOME_START, OUTCOME_END + 1))
    return peak >= EXPLOSION_RELATIVE * baseline


# ----------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------
def main():
    counts = load_counts(INPUT_CSV)
    # restrict to topics visible at freeze date (reduces noise from rare n-grams)
    freeze_counts = counts[counts[COL_YEAR] == FREEZE_END].groupby(COL_TOPIC)[COL_COUNT].sum()
    visible = set(freeze_counts[freeze_counts >= MIN_VISIBLE].index)
    topics = sorted(t for t in counts[COL_TOPIC].unique() if t in visible)
    print(f"Topics with >={MIN_VISIBLE} papers in {FREEZE_END}: {len(topics)}")

    rows = []
    for t in topics:
        s = series_for(counts, t)
        pos, crit = signature_positive(s)
        exp = exploded(s)
        rows.append({"topic": t, "predicted_positive": pos,
                     "exploded": exp, **crit})
    res = pd.DataFrame(rows)

    TP = int(((res.predicted_positive) & (res.exploded)).sum())
    FP = int(((res.predicted_positive) & (~res.exploded)).sum())
    FN = int(((~res.predicted_positive) & (res.exploded)).sum())
    TN = int(((~res.predicted_positive) & (~res.exploded)).sum())
    N = TP + FP + FN + TN

    precision = TP / (TP + FP) if (TP + FP) else float("nan")
    recall = TP / (TP + FN) if (TP + FN) else float("nan")
    base_rate = (TP + FN) / N if N else float("nan")

    fp_topics = res[res.predicted_positive & ~res.exploded].topic.tolist()
    fn_topics = res[~res.predicted_positive & res.exploded].topic.tolist()

    print("=" * 60)
    print("CONFUSION MATRIX (rows = 2021 prediction, cols = 2022-2025)")
    print(f"  TP={TP}  FP={FP}")
    print(f"  FN={FN}  TN={TN}   (N={N} topics)")
    print("-" * 60)
    print(f"  precision = {precision:.1%}")
    print(f"  recall    = {recall:.1%}")
    print(f"  base rate = {base_rate:.1%}")
    print("-" * 60)
    print(f"  false positives ({len(fp_topics)}): {fp_topics}")
    print(f"  false negatives ({len(fn_topics)}): {fn_topics}")
    print("=" * 60)

    # ready-to-paste LaTeX for the confusion table cells
    print("\n--- paste into Table~\\ref{tab:confusion} ---")
    print(f"Positive (all 4 criteria met) & {TP}~(TP) & {FP}~(FP) \\\\")
    print(f"Negative (criteria not met)   & {FN}~(FN) & {TN}~(TN) \\\\")
    print("\n--- paste into the validation sentence ---")
    print(f"precision of {precision*100:.0f}\\% ... recall of "
          f"{recall*100:.0f}\\% ... base explosion rate of "
          f"{base_rate*100:.0f}\\%")

    res.to_csv("backtest_per_topic.csv", index=False)
    print("\nPer-topic detail written to backtest_per_topic.csv")
    print("Inspect the false positives/negatives there and report them.")


if __name__ == "__main__":
    main()
