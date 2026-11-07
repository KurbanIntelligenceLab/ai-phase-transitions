# Legacy scripts (superseded)

These scripts produced the figures reported in the **first submission** of the paper
(27% precision / 63% recall against a 13.5% base rate on 223 topics). They count topics in
absolute units over an unconsolidated topic set. Following peer review, the analysis was
revised to use within-year shares, consolidated topic labels and equal-length trend windows.

| Legacy script | Replaced by |
|---|---|
| `backtest_signature.py` | `refreeze_thresholds.py` + `backtest_signature_revised.py` |
| `compute_popularity.py` | `compute_popularity_revised.py` |

They are kept for transparency only. Run from the repository root if needed
(e.g. `python legacy/backtest_signature.py`). The published results come from the revised
scripts in the root directory.
