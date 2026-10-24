# HBOS gap: verified tests and remaining uncertainty

Paper Table 1 HBOS AUCs are 0.9910 (breast-cancer), 0.9214 (pen-global), and 0.7651 (pen-local).
The main numeric port was corrected during review: it now counts final singleton observations and
uses consistent static-bin boundaries. Historical 2012 variants retain their historical behavior.
The table below comes from the new `hbos_gap_tests.csv`, not from the earlier stored output.

All best values are label-tuned oracles over the explicitly listed grid, not unbiased estimates or
proof that the authors used the same selection process. Standard AUC gives tied pairs half credit.
The optimistic metric gives every tied positive-negative pair full credit and is only a diagnostic bound.

| Dataset | Variant | Default AUC | Best AUC (oracle) | Best k (oracle) |
|---|---|---:|---:|---:|
| breast-cancer | T0 current code, dynamic | 0.9804 | 0.9863 | 150 |
| breast-cancer | T0 current code, static | 0.9829 | 0.9852 | 5 |
| breast-cancer | T1 July-2012 code, dynamic (k = values per bin) | 0.9804 | 0.9868 | 90 |
| breast-cancer | T2 July-2012 code, static (strict border, extra max bin) | 0.9829 | 0.9863 | 5 |
| breast-cancer | T0 current code, dynamic [T4 duplicates removed] | 0.9804 | 0.9863 | 150 |
| pen-global | T0 current code, dynamic | 0.7125 | 0.7801 | 15 |
| pen-global | T0 current code, static | 0.7146 | 0.7661 | 5 |
| pen-global | T1 July-2012 code, dynamic (k = values per bin) | 0.7125 | 0.7731 | 90 |
| pen-global | T2 July-2012 code, static (strict border, extra max bin) | 0.7222 | 0.7341 | 5 |
| pen-global | T0 current code, dynamic [T4 duplicates removed] | 0.7125 | 0.7801 | 15 |
| pen-local | T0 current code, dynamic | 0.6618 | 0.7292 | 400 |
| pen-local | T0 current code, static | 0.7202 | 0.7545 | 20 |
| pen-local | T1 July-2012 code, dynamic (k = values per bin) | 0.6618 | 0.7302 | 5 |
| pen-local | T2 July-2012 code, static (strict border, extra max bin) | 0.7011 | 0.7354 | 15 |
| pen-local | T0 current code, dynamic [T4 duplicates removed] | 0.6618 | 0.7292 | 400 |

The tested variants do not reproduce all three reported HBOS AUCs. Changing the static boundary
convention changes the results, so it would be incorrect to claim that all binning choices are ruled out.
Default neutral and optimistic AUCs agree for the tested variants; this does not rule out every possible
tie-handling effect at other settings. Perfect optimistic values in the historical dynamic variant are
tie artifacts, not perfect standard AUC. No exact duplicate feature rows occur in these three files.

The paper's complete preprocessing and optimal AUC parameter settings are unavailable in its text.
The current numeric Java conventions differ from the prose in Section 3, especially dynamic upper
edges. This review compared source code but did not execute the Java implementation within RapidMiner.
These are unresolved reproduction limits, not established explanations of a particular numerical gap.
Agreement by some other detectors does not prove that these files are identical to the paper's inputs.

The earlier ad hoc distinct-value-bin claim is omitted because the provided script does not reproduce it.
Sensitivity (`hbos_sensitivity.csv`) and timing (`timing.csv`, `timing_scaling.csv`) were re-run with the
corrected code on 24 Oct 2026. LOCI (`loci.csv`) does not use HBOS and was not re-run.
