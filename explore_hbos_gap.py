"""Test possible explanations for the gap between our HBOS AUC and Table 1 of the paper.

Each explanation is tested on its own (one change at a time) on all three data sets.
Writes results/hbos_gap_tests.csv. Labels are used only to compute AUC; the "best" column is an
oracle (label-tuned) number, "default" uses the sqrt(N) rule without looking at labels.

Usage: python explore_hbos_gap.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import data  # noqa: E402
from hbos import default_bins, hbos_scores  # noqa: E402
from hbos_2012 import hbos_scores_2012  # noqa: E402

KS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400]
PAPER = {"breast-cancer": 0.9910, "pen-global": 0.9214, "pen-local": 0.7651}


def aucs(y, s):
    """(neutral AUC with ties = 1/2, optimistic AUC with ties counted for the outliers)."""
    s = np.asarray(s, dtype=float)
    if np.isnan(s).any():
        raise ValueError("NaN scores cannot be evaluated")
    # Historical static-border behavior can yield infinity. Keep equal infinities
    # tied; this is diagnostic, not a repaired detector output.
    s = np.nan_to_num(s, posinf=1e300, neginf=-1e300)
    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    lo = np.searchsorted(neg, pos, side="left")
    hi = np.searchsorted(neg, pos, side="right")
    return roc_auc_score(y, s), hi.sum() / (len(pos) * len(neg))


VARIANTS = {
    "T0 current code, dynamic": lambda X, k: hbos_scores(X, bins=k, mode="dynamic"),
    "T0 current code, static": lambda X, k: hbos_scores(X, bins=k, mode="static"),
    "T1 July-2012 code, dynamic (k = values per bin)": lambda X, k: hbos_scores_2012(X, k, "dynamic"),
    "T2 July-2012 code, static (strict border, extra max bin)": lambda X, k: hbos_scores_2012(X, k, "static"),
}


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    rows = []
    for name in data.DATASETS:
        X, y = data.load(name)
        n = len(y)
        dup = n - len(np.unique(X, axis=0))
        print(f"{name}: n={n}, duplicate rows={dup}", flush=True)
        sets = {"": (X, y)}
        # T4: remove exact duplicate records (a possible preprocessing step)
        _, keep = np.unique(X, axis=0, return_index=True)
        keep.sort()
        sets[" [T4 duplicates removed]"] = (X[keep], y[keep])
        for suffix, (Xv, yv) in sets.items():
            for vname, fn in VARIANTS.items():
                if suffix and not vname.startswith("T0 current code, dynamic"):
                    continue
                res = []
                for k in KS:
                    a, ao = aucs(yv, fn(Xv, k))
                    res.append((k, a, ao))
                kd = default_bins(len(yv))
                a_d, ao_d = aucs(yv, fn(Xv, kd))
                best = max(res, key=lambda r: r[1])
                best_o = max(res, key=lambda r: r[2])
                rows.append(dict(dataset=name, test=vname + suffix, paper_auc=PAPER[name],
                                 default_k=kd, default_auc=round(a_d, 4), default_auc_optimistic=round(ao_d, 4),
                                 best_k_oracle=best[0], best_auc_oracle=round(best[1], 4),
                                 best_k_optimistic_oracle=best_o[0], best_auc_optimistic_oracle=round(best_o[2], 4),
                                 best_results_label_tuned=True))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "hbos_gap_tests.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
