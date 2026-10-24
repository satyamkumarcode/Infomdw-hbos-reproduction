"""Reproduce Table 1 (ROC-AUC) and Fig. 1 (AUC vs k) of Goldstein & Dengel (2012).

Usage:  python run_table1.py          (about 4 minutes; LOCI is in run_loci.py)
Writes results/sweep.csv, results/table1_comparison.csv, results/environment.json
and figures/fig1_auc_vs_k.png.

Labels are used ONLY to compute AUC. The "default" column uses no label-based choice.
The "best k" column picks, per method and data set, the k with the highest AUC: this is an
oracle (label-tuned) number; the paper's "optimal" selection procedure is not documented.
"""
import json
import importlib.metadata
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import data  # noqa: E402
import detectors as D  # noqa: E402
from hbos import default_bins  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
KS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]   # sweep for k / bins / clusters
CLUSTERS = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
DEFAULT_K = 10          # default neighbours for k-NN, LOF, COF, LoOP and default bins for PyOD HBOS
DEFAULT_CLUSTERS = 8    # default number of clusters for CBLOF / u-CBLOF

# Table 1 of the paper (AUC, "optimal parameter settings"); None = not computable
PAPER = {
    "HBOS": (0.9910, 0.9214, 0.7651), "k-NN": (0.9826, 0.9892, 0.9852),
    "LOF": (0.9916, 0.8864, 0.9878), "COF": (0.9888, 0.9586, 0.9688),
    "LoOP": (0.9882, 0.8492, 0.9864), "LOCI": (0.9678, 0.8868, None),
    "CBLOF": (0.8389, 0.6808, 0.7007), "u-CBLOF": (0.9743, 0.9923, 0.9767),
}

# method -> (function, parameter kind, feature scaling, default parameter, sweep)
METHODS = {
    "HBOS (authors, dynamic)": (D.hbos_dynamic, "bins", "raw", None, KS),
    "HBOS (authors, static)": (D.hbos_static, "bins", "raw", None, KS),
    "HBOS (PyOD)": (D.hbos_pyod, "bins", "raw", DEFAULT_K, KS),
    "k-NN": (D.knn_global, "k", "minmax", DEFAULT_K, KS),
    "LOF": (D.lof, "k", "minmax", DEFAULT_K, KS),
    "COF": (D.cof, "k", "minmax", DEFAULT_K, KS),
    "LoOP": (D.loop, "k", "minmax", DEFAULT_K, KS),
    "CBLOF": (lambda X, c: D.cblof(X, c, True), "clusters", "minmax", DEFAULT_CLUSTERS, CLUSTERS),
    "u-CBLOF": (lambda X, c: D.cblof(X, c, False), "clusters", "minmax", DEFAULT_CLUSTERS, CLUSTERS),
}
PAPER_KEY = {"HBOS (authors, dynamic)": "HBOS", "HBOS (authors, static)": "HBOS", "HBOS (PyOD)": "HBOS"}


def auc(y, score):
    score = np.asarray(score, dtype=float)
    if np.isnan(score).any():
        raise ValueError("NaN scores cannot be evaluated")
    score = np.nan_to_num(score, posinf=1e300, neginf=-1e300)
    return roc_auc_score(y, score)


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    rows = []
    for name in data.DATASETS:
        X, y = data.load(name)
        Xs = data.minmax(X)
        n = len(y)
        for method, (fn, kind, scaling, default, sweep) in METHODS.items():
            Z = X if scaling == "raw" else Xs
            for k in sweep:
                t0 = time.perf_counter()
                try:
                    a = auc(y, fn(Z, k))
                except ValueError as exc:
                    if method in ("CBLOF", "u-CBLOF") and "cluster" in str(exc).lower() and "separ" in str(exc).lower():
                        print(f"Skipped {name} {method} c={k}: {exc}", flush=True)
                        continue
                    raise
                rows.append(dict(dataset=name, n=n, method=method, param=kind, value=k, auc=a,
                                 seconds=time.perf_counter() - t0))
            if default is None:     # authors' HBOS: default rule is round(sqrt(N)) bins
                k = default_bins(n)
                rows.append(dict(dataset=name, n=n, method=method, param=kind, value=k,
                                 auc=auc(y, fn(Z, k)), seconds=np.nan, default=True))
            print(name, method, "done", flush=True)

    sweep = pd.DataFrame(rows)
    sweep["default"] = sweep.get("default", False)
    sweep["default"] = sweep["default"].fillna(False).astype(bool)
    # default flag for methods with a fixed default value
    for method, (_, _, _, default, _) in METHODS.items():
        if default is not None:
            sweep.loc[(sweep.method == method) & (sweep.value == default), "default"] = True
    sweep.to_csv(RES / "sweep.csv", index=False)

    # Table 1 comparison
    out = []
    for method in METHODS:
        paper = PAPER.get(PAPER_KEY.get(method, method))
        for i, name in enumerate(data.DATASETS):
            s = sweep[(sweep.dataset == name) & (sweep.method == method)]
            if s.empty:
                continue
            d = s[s["default"]]
            best = s.loc[s.auc.idxmax()]
            out.append(dict(dataset=name, method=method,
                            paper_auc=None if paper is None else paper[i],
                            default_value=None if d.empty else d.value.iloc[0],
                            default_auc=None if d.empty else round(d.auc.iloc[0], 4),
                            best_value=best.value, best_auc_oracle=round(best.auc, 4),
                            mean_auc_all_values=round(s.auc.mean(), 4)))
    comp = pd.DataFrame(out)
    comp.to_csv(RES / "table1_comparison.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(comp.to_string(index=False))

    (RES / "environment.json").write_text(json.dumps({
        "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__, "pyod": __import__("pyod").__version__,
        "ks": KS, "clusters": CLUSTERS, "default_k": DEFAULT_K, "default_clusters": DEFAULT_CLUSTERS,
        "packages": {p: importlib.metadata.version(p) for p in
                     ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "pyod", "PyNomaly", "numba", "threadpoolctl")}}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, name in zip(axes, data.DATASETS):
        for method in ["k-NN", "LOF", "COF", "LoOP", "HBOS (authors, dynamic)", "HBOS (authors, static)"]:
            s = sweep[(sweep.dataset == name) & (sweep.method == method) & (~sweep["default"] | (sweep.value.isin(KS)))]
            s = s[s.value.isin(KS)].sort_values("value")
            ax.plot(s.value, s.auc, marker=".", label=method)
        ax.set_title(name)
        ax.set_xlabel("k (neighbours / bins)")
    axes[0].set_ylabel("AUC")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_auc_vs_k.png", dpi=150)


if __name__ == "__main__":
    main()
