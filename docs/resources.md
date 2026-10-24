# Task 2 resources: HBOS reproduction

| Resource | Source | Licence | Status |
|---|---|---|---|
| Paper (Goldstein & Dengel, KI-2012 poster) | https://www.dfki.de/fileadmin/user_upload/import/6431_HBOS-KI-2012.pdf | n/a | not included (see link) |
| Authors' code (RapidMiner/Java plug-in) | https://github.com/Markus-Go/rapidminer-anomalydetection | AGPL-3.0 | used as reference, not included (clone from the link) |
| Benchmark data (Harvard Dataverse, published 6 Oct 2015) | https://doi.org/10.7910/DVN/OPQMVF | CC0 1.0 | downloaded into data/ (original CSV format, 21 Oct 2026) |
| Python HBOS + baselines | PyOD (HBOS, COF, CBLOF, LOCI), scikit-learn (LOF), PyNomaly (LoOP) | open source | `pip install -r requirements.txt` |

Data files needed (Dataverse): breast-cancer (77.1 KB), pen-global (65.8 KB), pen-local (545.3 KB), README.txt (4.3 KB).

Not implemented in this project: Fast-LOF, LDCOF, INFLO.
Folders: data/ src/ results/ figures/ tests/
