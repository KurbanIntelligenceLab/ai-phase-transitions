#!/usr/bin/env python3
"""
Robustness of the signature to the keyword source.

WHY THIS EXISTS
  Topic labels in topic_year_venue_counts.csv come from two sources:
  author-supplied keywords where the venue provides them (ICLR 2018+,
  NeurIPS 2021+, ICML 2025) and KeyBERT keyphrases extracted from titles
  elsewhere. The two vocabularies differ (e.g. "reinforcement learning"
  vs "deep reinforcement learning" / title trigrams), so a venue switching
  source produces artificial jumps in topic counts.

  This script measures every paper the same way: a paper counts for
  canonical topic k if its normalized title contains any surface variant
  of k as a whole-word phrase. It then re-freezes the thresholds on
  2017-2021 data (same procedure as refreeze_thresholds.py), re-runs the
  backtest, and scores the SAME topics evaluated in the main backtest.

OUTPUT
  robustness_keyword_source/topic_year_venue_counts.csv   title-based counts
  robustness_keyword_source/refrozen_thresholds.json      re-frozen thresholds
  robustness_keyword_source_results.txt                   summary
"""
import io, os, re, sys, shutil, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "robustness_keyword_source")
XLSX = os.path.join(HERE, "merged_papers_keywords_keybert.xlsx")
CACHE = os.path.join(OUTDIR, "papers_cache.pkl")
MIN_TOTAL = 10
os.makedirs(OUTDIR, exist_ok=True)


def load_bt(path, cwd=None):
    src = io.open(path, encoding="utf-8").read()
    src = src.replace('if __name__ == "__main__":\n    main()', "")
    old = os.getcwd()
    if cwd:
        os.chdir(cwd)
    bt = {}
    exec(compile(src, path, "exec"), bt)
    if cwd:
        os.chdir(old)
    return bt


def norm(s):
    s = str(s).lower().replace("-", " ").replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# 1. vocabulary: canonical topics of the main pipeline (>= MIN_TOTAL mentions)
bt0 = load_bt(os.path.join(HERE, "backtest_signature_revised.py"))
canon = bt0["canon"]
lab = pd.read_csv(os.path.join(HERE, "topic_year_venue_counts.csv"))
lab["c"] = lab["topic"].map(canon)
tot = lab.groupby("c")["count"].sum()
keep_c = set(tot[tot >= MIN_TOTAL].index)
phrase2c = {}
for t, c in lab[lab["c"].isin(keep_c)][["topic", "c"]].drop_duplicates().itertuples(index=False):
    p = norm(t)
    if p and len(p.split()) <= 5:
        phrase2c.setdefault(p, set()).add(c)
for c in keep_c:
    p = norm(c)
    if p and len(p.split()) <= 5:
        phrase2c.setdefault(p, set()).add(c)

# 2. title-based counts for every main-track paper
if os.path.exists(CACHE):
    df = pd.read_pickle(CACHE)
else:
    df = pd.read_excel(XLSX, usecols=["conference", "year", "paper_type", "title"])
    df.to_pickle(CACHE)
df = df[df["paper_type"].astype(str).str.lower() == "main"].copy()
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
rows = {}
for conf, yr, title in df[["conference", "year", "title"]].itertuples(index=False):
    w = norm(title).split()
    hit = set()
    for n in range(1, 6):
        for i in range(len(w) - n + 1):
            hit |= phrase2c.get(" ".join(w[i:i + n]), set())
    for c in hit:
        rows[(c, int(yr), conf)] = rows.get((c, int(yr), conf), 0) + 1
pd.DataFrame([(c, y, v, n) for (c, y, v), n in rows.items()],
             columns=["topic", "year", "venue", "count"]).to_csv(
    os.path.join(OUTDIR, "topic_year_venue_counts.csv"), index=False)

# 3. re-freeze thresholds on title-based data (pre-2022 only)
for f in ["refreeze_thresholds.py", "backtest_signature_revised.py"]:
    shutil.copy(os.path.join(HERE, f), os.path.join(OUTDIR, f))
subprocess.run([sys.executable, "refreeze_thresholds.py"], cwd=OUTDIR,
               check=True, capture_output=True)

# 4. score the same topics as the main backtest
bt = load_bt(os.path.join(OUTDIR, "backtest_signature_revised.py"), cwd=OUTDIR)
os.chdir(OUTDIR)
m = bt["load_shares"](bt["INPUT_CSV"])
os.chdir(HERE)
main_res = pd.read_csv(os.path.join(HERE, "backtest_per_topic_revised.csv"))
have = set(m["topic"].unique())
TP = FP = FN = TN = 0
surv_flag, surv_tp = [], []
for t, p_pos, p_ex in main_res[["topic", "predicted_positive", "exploded"]].itertuples(index=False):
    if t not in have:
        continue
    s = bt["series_for"](m, t)
    pos, _ = bt["signature_positive"](s)
    ex = bt["exploded"](s)
    TP += pos and ex; FP += pos and not ex; FN += (not pos) and ex; TN += (not pos) and (not ex)
    if p_pos and pos:
        surv_flag.append(t)
    if p_pos and p_ex and pos and ex:
        surv_tp.append(t)
N = TP + FP + FN + TN
lines = [
    "Robustness of the signature to keyword source (title-based counts)",
    f"topics scored (same as main backtest, present in title data): {N}",
    f"TP={TP} FP={FP} FN={FN} TN={TN}",
    f"precision={TP/max(TP+FP,1):.1%}  recall={TP/max(TP+FN,1):.1%}  base rate={(TP+FN)/N:.1%}",
    f"main-backtest flags still flagged: {len(surv_flag)} of {int(main_res.predicted_positive.sum())}: {sorted(surv_flag)}",
    f"main-backtest true positives still true positives: {sorted(surv_tp)}",
]
io.open(os.path.join(HERE, "robustness_keyword_source_results.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
