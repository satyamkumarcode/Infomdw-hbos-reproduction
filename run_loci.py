"""LOCI row of Table 1 (breast-cancer and pen-global only; the paper reports pen-local as not computable).

Usage: python run_loci.py      (breast-cancer about 4 minutes, pen-global about 27 hours)
Writes results/loci.csv.
"""
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import data  # noqa: E402
import detectors as D  # noqa: E402

PAPER = {"breast-cancer": 0.9678, "pen-global": 0.8868}


def main():
    rows = []
    for name in ["breast-cancer", "pen-global"]:
        X, y = data.load(name)
        t0 = time.perf_counter()
        a = roc_auc_score(y, D.loci(data.minmax(X)))
        rows.append(dict(dataset=name, method="LOCI", params="alpha=0.5, k=3, min-max scaled",
                         paper_auc=PAPER[name], auc=round(a, 4), seconds=round(time.perf_counter() - t0, 1)))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results" / "loci.csv", index=False)


if __name__ == "__main__":
    main()
