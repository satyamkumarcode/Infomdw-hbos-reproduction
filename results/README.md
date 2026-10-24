# Recorded experiment results

| File | Meaning | Producer |
|---|---|---|
| table1_comparison.csv | Paper, fixed-default and best-setting oracle AUC | run_table1.py |
| sweep.csv | Individual parameter settings and observed AUC/runtime | run_table1.py |
| hbos_gap_tests.csv | Historical/binning/tie/duplicate diagnostics; best values are oracle | explore_hbos_gap.py |
| hbos_sensitivity.csv | Bin rules and label-tuned per-feature/PyOD searches | sensitivity_hbos.py |
| loci.csv | Fixed-setting LOCI results on two datasets | run_loci.py |
| timing.csv | Repeated pen-global timing; k-means is an LDCOF proxy | run_timing.py |
| timing_scaling.csv | One timing per synthetic size/method | run_timing.py |
| environment.json | Main Table 1 execution environment | run_table1.py |
| review_parameters.json | Effective defaults and input hashes captured during review | Review snapshot |

Each CSV is a separate recorded experiment. The Table 1 environment metadata does not
establish the environment or code revision of every later run. File modification times
are inventory metadata, not proof of execution provenance. A manifest records the exact
distributed bytes. Preserve oracle labels and do not interpret diagnostic optimistic AUC
as standard ROC-AUC. See docs/parameters.md for grids and implementation choices.
