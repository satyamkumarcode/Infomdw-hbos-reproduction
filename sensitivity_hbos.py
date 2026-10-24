"""Sensitivity analysis: how high can HBOS get with other bin settings?

ORACLE ANALYSIS. Every "best" value below is chosen with the labels, so it is an optimistic upper bound
of what HBOS can reach on these files, not an unbiased result. It answers one question: is there any
reasonable HBOS setting that reaches the paper's Table 1 values?

 S1  textbook bin-count rules (no labels used): sqrt, Sturges, Rice, Scott, Freedman-Diaconis
 S2  per-feature bin counts: greedy search over k in GRID for every feature separately (label-tuned)
 S3  PyOD HBOS: n_bins (including "auto") x alpha x tol grid (label-tuned)

Usage: python sensitivity_hbos.py      (a few minutes)
Writes results/hbos_sensitivity.csv.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import data  # noqa: E402
from hbos import _feature_scores  # noqa: E402

PAPER = {"breast-cancer": 0.9910, "pen-global": 0.9214, "pen-local": 0.7651}
GRID = [2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400]


def auc(y, s):
    return roc_auc_score(y, np.nan_to_num(s, nan=0.0, posinf=1e300))


def rule_bins(col, rule):
    """Number of bins for one feature by a textbook rule (falls back to sqrt(N) if a rule degenerates)."""
    n = len(col)
    span = col.max() - col.min()
    sqrt_k = round(math.sqrt(n))
    if rule == "sqrt(N) (paper)":
        return sqrt_k
    if rule == "Sturges":
        return math.ceil(math.log2(n)) + 1
    if rule == "Rice":
        return math.ceil(2 * n ** (1 / 3))
    if rule == "Scott":
        h = 3.49 * col.std() * n ** (-1 / 3)
    else:  # Freedman-Diaconis
        q75, q25 = np.percentile(col, [75, 25])
        h = 2 * (q75 - q25) * n ** (-1 / 3)
    return sqrt_k if h <= 0 or span == 0 else max(1, math.ceil(span / h))


def main():
    rows = []
    for name in data.DATASETS:
        X, y = data.load(name)
        d = X.shape[1]

        # S1: textbook rules, per feature, both modes (no labels involved in the choice)
        for rule in ["sqrt(N) (paper)", "Sturges", "Rice", "Scott", "Freedman-Diaconis"]:
            ks = [rule_bins(X[:, j], rule) for j in range(d)]
            for mode in ("dynamic", "static"):
                s = sum(_feature_scores(X[:, j], ks[j], mode) for j in range(d))
                rows.append(dict(dataset=name, analysis="S1 bin rule", setting=f"{rule}, {mode}",
                                 label_tuned=False, auc=round(auc(y, s), 4), paper_auc=PAPER[name],
                                 detail=f"median bins {int(np.median(ks))}"))

        # S2: per-feature bin counts, greedy coordinate ascent on AUC (label-tuned)
        for mode in ("dynamic", "static"):
            F = {(j, k): _feature_scores(X[:, j], k, mode) for j in range(d) for k in GRID}
            best_k = max(GRID, key=lambda k: auc(y, sum(F[(j, k)] for j in range(d))))
            ks = [best_k] * d
            cur = auc(y, sum(F[(j, ks[j])] for j in range(d)))
            for _ in range(3):  # passes over all features
                improved = False
                for j in range(d):
                    base = sum(F[(i, ks[i])] for i in range(d) if i != j)
                    k_j = max(GRID, key=lambda k: auc(y, base + F[(j, k)]))
                    a = auc(y, base + F[(j, k_j)])
                    if a > cur + 1e-12:
                        ks[j], cur, improved = k_j, a, True
                if not improved:
                    break
            rows.append(dict(dataset=name, analysis="S2 per-feature bins (oracle)", setting=mode,
                             label_tuned=True, auc=round(cur, 4), paper_auc=PAPER[name],
                             detail=f"start k={best_k}; bins per feature {ks}"))

        # S3: PyOD HBOS grid (label-tuned)
        from pyod.models.hbos import HBOS
        best = (-1, None)
        for nb in ["auto", 5, 10, 20, 30, 50, 100]:
            for alpha in (0.01, 0.05, 0.1, 0.2, 0.5):
                for tol in (0.1, 0.5, 0.9):
                    a = auc(y, HBOS(n_bins=nb, alpha=alpha, tol=tol).fit(X).decision_scores_)
                    if a > best[0]:
                        best = (a, f"n_bins={nb}, alpha={alpha}, tol={tol}")
        rows.append(dict(dataset=name, analysis="S3 PyOD grid (oracle)", setting=best[1], label_tuned=True,
                         auc=round(best[0], 4), paper_auc=PAPER[name], detail="105 settings"))
        print(name, "done", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "hbos_sensitivity.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(out.drop(columns="detail").to_string(index=False))
        print(out[out.analysis.str.startswith("S2")][["dataset", "setting", "detail"]].to_string(index=False))


if __name__ == "__main__":
    main()
