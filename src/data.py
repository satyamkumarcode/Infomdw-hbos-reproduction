"""Loading of the three benchmark data sets (Harvard Dataverse, doi:10.7910/DVN/OPQMVF)."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASETS = ["breast-cancer", "pen-global", "pen-local"]


def load(name):
    """Return (X, y). y = 1 for outlier ("o"), 0 for normal ("n").

    The label column is excluded from X and never passed to detector fitting.
    Runners compute AUC from y; explicitly marked oracle searches use that AUC
    to select parameters, including per-feature bin counts.
    """
    df = pd.read_csv(DATA_DIR / f"{name}-unsupervised-ad.tab", header=None)
    y = (df.iloc[:, -1].str.strip('"') == "o").astype(int).to_numpy()
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    return X, y


def minmax(X):
    """Scale each feature to [0, 1] (constant features become 0)."""
    span = X.max(axis=0) - X.min(axis=0)
    span[span == 0] = 1.0
    return (X - X.min(axis=0)) / span
