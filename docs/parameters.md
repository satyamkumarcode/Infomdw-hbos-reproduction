# Experiment parameters and implementation notes

## Main experiment

| Parameter | Value |
|---|---|
| HBOS mode | dynamic bin width (paper's recommendation) and static bin width |
| HBOS bins k | default `round(sqrt(N))` = 19 / 28 / 82; sweep 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100 |
| HBOS score | `sum over features of log10(1 / normalised bin height)`, max bin height per feature = 1 |
| HBOS input | raw feature values (HBOS is scale-invariant per feature) |
| PyOD HBOS | `HBOS(n_bins=k)`, other arguments default (`alpha=0.1, tol=0.5`); default k = 10 |
| Scaling for all other methods | min-max to [0, 1] per feature |
| k-NN | distance to the k-th nearest neighbour, Euclidean; default k = 10, same sweep as above |
| LOF | scikit-learn `LocalOutlierFactor(n_neighbors=k)`; default k = 10, sweep |
| COF | PyOD `COF(n_neighbors=k, method="fast")`; default k = 10, sweep |
| LoOP | PyNomaly `LocalOutlierProbability(n_neighbors=k, extent=3)`; default k = 10, sweep |
| CBLOF / u-CBLOF | PyOD `CBLOF(n_clusters=c, alpha=0.9, beta=5, use_weights=True/False, random_state=0, contamination=0.1)`; default c = 8, sweep 2, 3, 4, 5, 6, 8, 10, 15, 20, 30 (values PyOD rejects are skipped) |
| LOCI | PyOD `LOCI(alpha=0.5, k=3)`; breast-cancer and pen-global only |
| AUC | scikit-learn `roc_auc_score` (ties count ½) |
| "Best k" | k with the highest AUC per method and data set: **label-tuned (oracle)**, not an unbiased estimate; the paper does not disclose whether its "optimal" settings used the same selection procedure |
| Timing | single thread, mean of 1,000 runs on pen-global after one warm-up run; scaling: one run on N(0,1) data with 15 features, seed 0 |
| Random seed | 0 for CBLOF/k-means and synthetic timing-data generation |


### Additional experiment settings

- `explore_hbos_gap.py`: k = 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400.
  Tests the current dynamic/static port, the July-2012 dynamic/static port, optimistic tie handling,
  and removal of exact duplicate feature rows. Default k uses rounded sqrt(N) after deduplication.
  In the 2012 dynamic variant, k means target records per bin, not bin count. The optimistic metric
  awards every tied positive-negative pair to the positive; it is a diagnostic bound, not standard AUC.
  Each best result selects its k using labels (oracle).
- `sensitivity_hbos.py`, S1: sqrt(N), Sturges, Rice, Scott and Freedman-Diaconis bin counts, calculated
  per feature for both dynamic and static modes. Degenerate width rules fall back to rounded sqrt(N).
  Scott uses 3.49 times the population standard deviation times N^(-1/3); Freedman-Diaconis uses
  twice the interquartile range times N^(-1/3). Counts round upward except the sqrt rule.
- S2: per-feature k grid = 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400.
  Initial common k maximises AUC; greedy coordinate ascent uses at most 3 passes and accepts an
  improvement greater than 1e-12. Both modes are tested. This entire search is label-tuned.
- S3: PyOD n_bins = auto, 5, 10, 20, 30, 50, 100; alpha = 0.01, 0.05, 0.1, 0.2, 0.5;
  tol = 0.1, 0.5, 0.9 (105 combinations). The reported maximum is label-tuned.
- Timing: `--repeats` defaults to 1,000, compared with 10,000 in the paper. HBOS uses raw data;
  other methods use min-max scaling on pen-global. The k-means proxy uses 30 clusters, n_init=1,
  max_iter=100, random_state=0, with early stopping permitted. Synthetic scaling uses unscaled
  standard-normal features at N = 10,000, 100,000, 1,000,000, 15 features and seed 0; k-NN and LOF
  stop at 100,000. Each synthetic setting is timed once.
- Library defaults not overridden by the code remain in force. For example, CBLOF uses its default
  clustering estimator, and neighbour methods use their libraries' default search algorithms.
  Requirements specify version ranges, not a locked environment; `results/environment.json` records
  versions from `run_table1.py`. Results and runtimes may differ across installed versions.

The review also records full effective estimator defaults in `results/review_parameters.json`
and exact installed versions in `requirements-review-lock.txt`. To recreate that reviewed environment,
install with `python -m pip install -r requirements-review-lock.txt` in a fresh environment.

## Not reproduced

### Corrections to inherited Java defects

The main `src/hbos.py` is now a corrected numeric port, not bit-for-bit Java compatibility.
The existing `HBOS (authors, ...)` result labels identify its origin, not identical behavior.
Dynamic construction now counts a final singleton (the Java loop could omit it); static counting
now uses the same half-open intervals as score lookup, including the maximum in the last bin.
Constant features and single-record inputs contribute zero, and invalid arrays/bin counts are rejected.
The 2012 diagnostic retains its historical binning, including the omitted-tail behavior, for comparison.
NaN scores now fail evaluation rather than silently becoming zero. Infinite historical scores remain
tied at a large sentinel for diagnostic AUC; they do not establish a successful detector result.

Other Java conventions remain: dynamic upper edges use the next group's first value, whereas
Section 3 describes the group's last value; zero-width final groups merge into the previous bin.
Only numeric features are supported. Both modes sort their inputs, so this implementation's static
construction is not the paper's linear-time construction. Base-10 logs preserve the ranking.
Java's extra per-feature density scale cancels under peak normalisation and is omitted.

Results have separate execution histories. environment.json records the Table 1 environment, not necessarily every other run. File timestamps alone do not identify the code revision used. The original audit reran Table 1 and gap analysis; sensitivity and timing files were subsequently updated. Best-result gap columns carry `_oracle` suffixes. See results/README.md for provenance limits.

Fast-LOF, LDCOF and INFLO are not implemented in this project, so their Table 1 rows are not reproduced. For timing, a plain k-means with 30 centroids and
at most 100 iterations stands in for LDCOF, but this proxy omits LDCOF scoring and is not an LDCOF runtime reproduction. LOCI is not run on pen-local: the
paper reports it as not computable. The 1,000,000-record timing data set of the paper is not public.

The Python port has not been checked against scores produced by running the authors' Java code (that needs the
RapidMiner libraries); it was checked against hand-computed examples (`tests/`).
