"""Run-time comparison of Section 4 of the paper.

Part 1: mean run time on pen-global over repeated runs (the paper used 10,000 repeats on one thread).
Part 2: scaling on synthetic data with 15 features (the paper's 1,000,000-record data set is not public).

Usage: python run_timing.py [--repeats 1000]
Writes results/timing.csv and results/timing_scaling.csv.
Single-threaded: OMP/BLAS threads are limited to 1 before numpy and scikit-learn are imported.
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"

import argparse  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import data  # noqa: E402
import detectors as D  # noqa: E402

# Paper, Section 4: mean time on pen-global (ms)
PAPER_MS = {"HBOS (authors, dynamic)": 3.8, "HBOS (authors, static)": 4.1, "k-NN": 28.5, "LOF": 28.0,
            "k-means, 30 centroids, 100 iterations": 30.0}


def kmeans_score(X):
    """Distance to the nearest of 30 k-means centroids (one start, 100 iterations).

    LDCOF is not available in Python; this measures the k-means cost that dominates LDCOF's run time.
    """
    km = KMeans(n_clusters=30, n_init=1, max_iter=100, random_state=0).fit(X)
    return km.transform(X).min(axis=1)


METHODS = {
    "HBOS (authors, dynamic)": lambda X: D.hbos_dynamic(X),
    "HBOS (authors, static)": lambda X: D.hbos_static(X),
    "HBOS (PyOD, 10 bins)": lambda X: D.hbos_pyod(X, 10),
    "k-NN": lambda X: D.knn_global(X, 10),
    "LOF": lambda X: D.lof(X, 10),
    "k-means, 30 centroids, 100 iterations": kmeans_score,
}


def mean_ms(fn, X, repeats):
    fn(X)  # warm-up (imports, caches)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(X)
    return (time.perf_counter() - t0) / repeats * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1000)
    args = ap.parse_args()

    X, _ = data.load("pen-global")
    Xs = data.minmax(X)
    rows = []
    for name, fn in METHODS.items():
        ms = mean_ms(fn, X if name.startswith("HBOS") else Xs, args.repeats)
        rows.append(dict(method=name, dataset="pen-global", repeats=args.repeats, mean_ms=round(ms, 2),
                         paper_ms=PAPER_MS.get(name)))
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    hb = t.loc[t.method == "HBOS (authors, dynamic)", "mean_ms"].iloc[0]
    t["times_slower_than_hbos_dynamic"] = (t.mean_ms / hb).round(2)
    t["cpu"] = platform.processor()
    t.to_csv(ROOT / "results" / "timing.csv", index=False)

    # Part 2: scaling (one run each; LOF and k-NN are stopped at 100,000 records)
    rng = np.random.default_rng(0)
    rows = []
    for n in (10_000, 100_000, 1_000_000):
        Z = rng.normal(size=(n, 15))
        for name in ("HBOS (authors, dynamic)", "HBOS (authors, static)", "k-NN", "LOF"):
            if name in ("k-NN", "LOF") and n > 100_000:
                continue
            t0 = time.perf_counter()
            METHODS[name](Z)
            rows.append(dict(method=name, n=n, features=15, seconds=round(time.perf_counter() - t0, 3)))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "timing_scaling.csv", index=False)


if __name__ == "__main__":
    main()
