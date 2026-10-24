"""Competitor detectors from Table 1. Every function returns outlier scores (higher = more anomalous)."""
import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from hbos import hbos_scores


def hbos_dynamic(X, k=None):
    return hbos_scores(X, bins=k, mode="dynamic")


def hbos_static(X, k=None):
    return hbos_scores(X, bins=k, mode="static")


def hbos_pyod(X, k=10):
    from pyod.models.hbos import HBOS
    return HBOS(n_bins=k).fit(X).decision_scores_


def knn_global(X, k):
    """Distance to the k-th nearest neighbour (Ramaswamy et al. 2000)."""
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return dist[:, k]


def lof(X, k):
    return -LocalOutlierFactor(n_neighbors=k).fit(X).negative_outlier_factor_


def cof(X, k):
    from pyod.models.cof import COF
    return COF(n_neighbors=k, method="fast").fit(X).decision_scores_


def loop(X, k):
    from PyNomaly import loop as pyn
    model = pyn.LocalOutlierProbability(X, n_neighbors=k, extent=3).fit()
    return np.asarray(model.local_outlier_probabilities, dtype=float)


def cblof(X, c, use_weights=True):
    from pyod.models.cblof import CBLOF
    model = CBLOF(n_clusters=c, alpha=0.9, beta=5, use_weights=use_weights,
                  random_state=0, contamination=0.1)
    return model.fit(X).decision_scores_


def loci(X):
    from pyod.models.loci import LOCI
    return LOCI(alpha=0.5, k=3).fit(X).decision_scores_
