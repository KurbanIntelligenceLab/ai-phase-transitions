"""Sensitivity of the out-of-sample evaluation to the explosion threshold
and to the visibility floor. Answers reviewer round-2 methodological point."""
import io, sys
import pandas as pd, numpy as np

src = io.open("backtest_signature_revised.py", encoding="utf-8").read()
src = src.replace('if __name__ == "__main__":\n    main()', "")
bt = {}
exec(compile(src, "bt", "exec"), bt)

load_shares = bt["load_shares"]
series_for = bt["series_for"]
share_at = bt["share_at"]
signature_positive = bt["signature_positive"]
FREEZE_END = bt["FREEZE_END"]
OUTCOME_START = bt["OUTCOME_START"]
OUTCOME_END = bt["OUTCOME_END"]
MIN_VISIBLE = bt["MIN_VISIBLE"]
INPUT_CSV = bt["INPUT_CSV"]

m = load_shares(INPUT_CSV)
fz = m[m["year"] == FREEZE_END].groupby("topic")["share"].sum()


def evaluate(mult, minvis):
    visible = set(fz[fz >= minvis].index)
    topics = sorted(t for t in m["topic"].unique() if t in visible)
    TP = FP = FN = TN = 0
    for t in topics:
        s = series_for(m, t)
        pos, _ = signature_positive(s)
        base = share_at(s, FREEZE_END)
        if base == 0:
            ex = False
        else:
            peak = max(share_at(s, y) for y in range(OUTCOME_START, OUTCOME_END + 1))
            ex = peak >= mult * base
        if pos and ex:
            TP += 1
        elif pos and not ex:
            FP += 1
        elif (not pos) and ex:
            FN += 1
        else:
            TN += 1
    N = TP + FP + FN + TN
    prec = TP / (TP + FP) if (TP + FP) else float("nan")
    rec = TP / (TP + FN) if (TP + FN) else float("nan")
    br = (TP + FN) / N if N else float("nan")
    return dict(N=N, TP=TP, FP=FP, FN=FN, TN=TN, prec=prec, rec=rec,
                base=br, lift=prec / br if br else float("nan"))


print("=== A. explosion multiplier sweep (visibility floor held at published value) ===")
print(f"{'mult':>6} {'N':>5} {'base%':>7} {'TP':>4} {'FP':>4} {'FN':>4} {'prec%':>7} {'rec%':>7} {'lift':>6}")
rowsA = []
for mult in [1.2, 1.3, 1.4, 1.5, 1.6, 1.75, 2.0, 2.5, 3.0]:
    r = evaluate(mult, MIN_VISIBLE)
    rowsA.append((mult, r))
    print(f"{mult:>6} {r['N']:>5} {r['base']*100:>7.1f} {r['TP']:>4} {r['FP']:>4} "
          f"{r['FN']:>4} {r['prec']*100:>7.1f} {r['rec']*100:>7.1f} {r['lift']:>6.2f}")

recs = [r["rec"] for _, r in rowsA]
lifts = [r["lift"] for _, r in rowsA]
print(f"\nrecall range: {min(recs)*100:.1f}% - {max(recs)*100:.1f}%")
print(f"lift   range: {min(lifts):.2f} - {max(lifts):.2f}")

print("\n=== B. visibility-floor sweep (explosion multiplier held at 1.5) ===")
print(f"{'floor x':>8} {'N':>5} {'base%':>7} {'prec%':>7} {'rec%':>7} {'lift':>6}")
for k in [0.5, 0.75, 1.0, 1.5, 2.0]:
    r = evaluate(1.5, MIN_VISIBLE * k)
    print(f"{k:>8} {r['N']:>5} {r['base']*100:>7.1f} {r['prec']*100:>7.1f} "
          f"{r['rec']*100:>7.1f} {r['lift']:>6.2f}")

print("\n=== C. LaTeX rows for sensitivity table (panel A) ===")
for mult, r in rowsA:
    star = r"$^{\dagger}$" if abs(mult - 1.5) < 1e-9 else ""
    print(f"{mult:.2f}{star} & {r['base']*100:.1f} & {r['TP']} & {r['FP']} & {r['FN']} "
          f"& {r['prec']*100:.1f} & {r['rec']*100:.1f} & {r['lift']:.2f} " + r"\\")
