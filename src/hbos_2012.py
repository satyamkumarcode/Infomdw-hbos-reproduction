# Derived from DFKI RapidMiner Anomaly Detection Extension.
# Copyright 2009-2012 DFKI or its licensors.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Python adaptation and corrections: Satyam Kumar, 2026. See NOTICE.md.
"""HBOS as implemented in the authors' July-2012 commit c49d3fb (closest to the paper).

Differences to the later code in hbos.py (checked in HistogramEvaluator.java, c49d3fb):
 - dynamic mode: the parameter is the number of VALUES PER BIN, not the number of bins
   (changed in commit 9dcf17e, Sept 2013). At k = round(sqrt(N)) the targets are close, but need not coincide:
   the later version uses floor(N/k) values per bin.
 - static mode: bins are half-open [from, to) also when counting, so no border quirk, and the
   maximum value is not part of the k-th bin: it ends up in an extra (k+1)-th bin.
"""
import numpy as np

from hbos import _dynamic_bins_per_bin, _scores_from_bins, default_bins


def _static_bins_2012(x, k):
    n = len(x)
    width = (x[-1] - x[0]) / k
    if width == 0:
        width = 1.0
    bins = []
    first = 0
    start = x[0]
    while first < n:
        stop = start + width
        upto = int(np.searchsorted(x, stop, side="left"))  # strict: value < stop
        bins.append([start, stop, upto - first])
        first = upto
        start = stop
    return bins


def hbos_scores_2012(X, k=None, mode="dynamic"):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    k = default_bins(n) if k is None else int(k)
    score = np.zeros(n)
    for j in range(d):
        x = np.sort(X[:, j])
        bins = _dynamic_bins_per_bin(x, k) if mode == "dynamic" else _static_bins_2012(x, k)
        score += _scores_from_bins(X[:, j], bins)
    return score
