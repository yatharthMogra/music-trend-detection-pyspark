# Music Trend Detection Pipeline

Scalable PySpark pipeline (CS-GY 6513 Big Data): breakout signal detection from listening events, aligned with `big_data_pipeline_spec.md`.

## Branches

- **`main`**: stable releases (merge from `develop` after integration).
- **`develop`**: integration branch; open PRs here from `feature/*` branches.
- **`feature/<name>`**: one branch per contributor; branch from latest `develop`, rebase/merge before PR.

## Run order (JupyterHub)

Run notebooks **in order** on the cluster; each stage reads/writes HDFS under your user.

| Step | Notebook |
|------|----------|
| 1 | `01_environment_and_data.ipynb` |
| 2 | `02_ingestion_parquet.ipynb` |
| 3 | `03_feature_engineering.ipynb` |
| 4 | `04_streaming.ipynb` |
| 5 | `05_model_training.ipynb` |
| 6 | `06_dashboard.ipynb` |

## HDFS contract

All paths are rooted at:

`hdfs:///user/<YOUR_UNIX_USER>/music/`

Shared constants live in [`hdfs_paths.py`](hdfs_paths.py). Notebooks prepend the repo root to `sys.path` and import `hdfs_paths`.

Layout:

- `raw/` — Last.fm, Spotify charts, Billboard, MSD audio (CSV)
- `processed/` — Parquet: `events/`, `spotify/`, `billboard/`, `audio/`, `features/`
- `streaming/` — `input/`, `output/`, `checkpoint/`
- `models/` — `rf_model/` (saved MLlib pipeline)

## Spark settings

- `spark.sql.shuffle.partitions` = **50** (cluster-sized; do not leave default 200).
- Use **explicit CSV schemas** in ingestion cells.

## Datasets

Sources and manual download fallbacks are documented in notebook 01 (Kaggle may require auth).

## Requirements

NYU JupyterHub image: PySpark 3.x, HDFS CLI. Notebook 06 may use `pip install scikit-learn --user` for ROC plotting only.
