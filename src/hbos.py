# Derived from DFKI RapidMiner Anomaly Detection Extension.
# Copyright 2009-2012 DFKI or its licensors.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Python adaptation and corrections: Anonymous contributor, 2026. See NOTICE.md.
"""HBOS (Goldstein & Dengel, 2012), ported from the authors' Java code.

Reference: https://github.com/Markus-Go/rapidminer-anomalydetection, files src/de/dfki/madm/
anomalydetection/evaluator/statistical_based/HistogramEvaluator.java and HistogramBin.java.
Comments marked [Java: ...] name the Java method that a block reproduces.
Intentional corrections: count a final singleton in dynamic histograms and give
constant features zero scores, including one-record inputs; count static border
values in the same bin used for scoring. The historical
helper keeps the Java tail behavior unless fix_tail=True is requested.
"""
import math

import numpy as np


def default_bins(n):
    """[Java: evaluate] default number of bins is round(sqrt(N))."""
    return int(math.floor(math.sqrt(n) + 0.5))


def _dynamic_bins(x, k):
    """[Java: createDynamicHistogram] equal-count bins on sorted values x.

    Returns bins as [start, end, quantity]. A bin takes floor(N/k) values; equal
    values never straddle a border, and a value that occurs more than floor(N/k)
    times gets a bin of its own.
    """
    return _dynamic_bins_per_bin(x, len(x) // k, fix_tail=True)


def _dynamic_bins_per_bin(x, per_bin, fix_tail=False):
    """Build bins with a supplied occupancy target.

    fix_tail=False preserves the authors' omitted-final-singleton behavior for
    the historical 2012 comparison. The main implementation corrects it.
    """
    n = len(x)
    bins = []
    first = 0
    while first < (n if fix_tail else n - 1):
        end = min(first + per_bin, n)
        last = first
        qty = 1
        for i in range(first + 1, end):
            if x[i] == x[last]:
                qty += 1
                last += 1
            elif i + per_bin < n and x[i] == x[i + per_bin]:
                break  # [Java: check_amount] a long run starts here, give it its own bin
            else:
                qty += 1
                last += 1
        i = last + 1
        while i < n and x[i] == x[last]:  # never split equal values
            qty += 1
            last += 1
            i += 1
        stop = x[last + 1] if last + 1 < n else x[n - 1]
        if stop - x[first] > 0:
            bins.append([x[first], stop, qty])
        elif not bins:
            bins.append([x[first], stop + 1.0, qty])
        else:  # a zero-width bin at the end is merged into the previous bin
            bins[-1][2] += qty
            bins[-1][1] = stop
        first = last + 1
    return bins


def _static_bins(x, k):
    """[Java: createStaticHistogram] k equal-width bins between min and max."""
    n = len(x)
    if x[0] == x[-1]:
        # Unlike Java's k=1 branch, never construct a zero-width constant bin.
        return [[x[0], x[0] + 1.0, n]]
    with np.errstate(over="ignore", under="ignore"):
        width = (x[-1] - x[0]) / k
    if not np.isfinite(width) or width <= 0:
        raise ValueError("bin width is not representable; rescale data or reduce bins")
    bins = []
    first = 0
    start = x[0]
    while first < n:
        last_bin = len(bins) == k - 1
        stop = x[-1] if last_bin else start + width
        if stop <= start or not np.isfinite(stop - start):
            raise ValueError("bin boundaries are not representable; rescale data or reduce bins")
        # Correction to Java: use the same half-open boundaries as score lookup.
        # The last bin includes the maximum, matching Java's final-bin lookup.
        upto = int(np.searchsorted(x, stop, side="right" if last_bin else "left"))
        bins.append([start, stop, upto - first])
        first = upto
        start = start + width
    return bins


def _feature_scores(col, k, mode):
    """Per-record score log10(1/height) of one feature."""
    if np.all(col == col[0]):
        # A single occupied bin has peak-normalised height one in either mode.
        return np.zeros(len(col))
    x = np.sort(col)
    bins = _dynamic_bins(x, k) if mode == "dynamic" else _static_bins(x, k)
    return _scores_from_bins(col, bins)


def _scores_from_bins(col, bins):
    """Score every value of one feature given its histogram bins [start, end, quantity]."""
    starts = np.array([b[0] for b in bins], dtype=float)
    stops = np.array([b[1] for b in bins], dtype=float)
    qty = np.array([b[2] for b in bins], dtype=float)
    height = qty / (stops - starts)          # [Java: calc_score] density = count / width
    rel = height / height.max()              # [Java: normalize_score] max height -> 1.0
    with np.errstate(divide="ignore"):
        bin_score = np.log10(1.0 / rel)      # empty bins give +inf
    # [Java: get_score] Lookup uses [from, to), with the final maximum included.
    idx = np.searchsorted(starts, col, side="right") - 1
    return bin_score[idx]


def hbos_scores(X, bins=None, mode="dynamic"):
    """HBOS outlier scores (higher = more anomalous).

    X: (n_samples, n_features) numeric array. bins: k, default round(sqrt(N)).
    mode: "dynamic" (equal-count bins) or "static" (equal-width bins).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("X must be a nonempty 2D array with at least one feature")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    if bins is not None and (
        isinstance(bins, (bool, np.bool_))
        or not isinstance(bins, (int, np.integer))
        or bins <= 0
    ):
        raise ValueError("bins must be a positive integer")
    n, d = X.shape
    k = default_bins(n) if bins is None else int(bins)
    if mode not in ("dynamic", "static"):
        raise ValueError("mode must be 'dynamic' or 'static'")
    score = np.zeros(n)
    for j in range(d):  # features are treated as independent: scores add up
        score += _feature_scores(X[:, j], k, mode)
    return score
