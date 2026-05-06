# Music Trend Detection Pipeline

Scalable PySpark pipeline (CS-GY 6513 Big Data): breakout signal detection from listening events, aligned with `big_data_pipeline_spec.md`.

## Branches

- **`main`**: stable releases (merge from `develop` after integration).
- **`develop`**: integration branch; open PRs here from `feature/*` branches.
- **`feature/<name>`**: one branch per contributor; branch from latest `develop`, rebase/merge before PR.

## Run order (JupyterHub / local PySpark)

Run notebooks **in order**; each stage reads/writes project-local data paths by default.

| Step | Notebook |
|------|----------|
| 1 | `01_environment_and_data.ipynb` |
| 2 | `02_ingestion_parquet.ipynb` |
| 3 | `03_feature_engineering.ipynb` |
| 4 | `04_streaming.ipynb` |
| 5 | `05_model_training.ipynb` |
| 6 | `06_dashboard.ipynb` |
| 7 | `07_project_report.ipynb` (architecture & report; no Spark required) |

## What each file does (mini simulation)

Use this as a quick "input -> process -> output" map for new readers.

### `hdfs_paths.py` (shared config)

- **Purpose:** Central path constants and Spark shuffle setting used by all notebooks.
- **Inputs:** Optional env vars `BDCAP_PROJECT_ROOT`, `BDCAP_SHUFFLE_PARTITIONS`.
- **Outputs:** Resolved locations like `data/raw/...`, `data/processed/...`, `data/models/rf_model`.
- **Example:** If `BDCAP_PROJECT_ROOT=/repo`, then `PROCESSED_FEATURES=/repo/data/processed/features`.

### `01_environment_and_data.ipynb` (setup + raw data)

- **Purpose:** Verify Spark/filesystem access and populate raw dataset files.
- **Inputs:** URLs + manual downloads placed in `_downloads/`:
  - `hetrec2011-lastfm-2k.zip` (auto-download)
  - `charts.csv`, `Hot 100.csv`, `msd_audio_features.csv` (usually manual)
- **Outputs:** Raw files copied into `data/raw/...`; schema previews and row counts printed.
- **Mini simulation output:** `Spark version: 3.x`, `Directories ensured.`, `=== Spotify === ... rows: 527172`.

### `02_ingestion_parquet.ipynb` (raw -> Parquet)

- **Purpose:** Clean raw CSV/TSV and write analytics-ready Parquet.
- **Inputs:** `RAW_LASTFM`, `RAW_SPOTIFY`, `RAW_BILLBOARD`, `RAW_MSD`.
- **Processing note:** Last.fm `(userID, artistID, weight)` is expanded into synthetic timestamped events.
- **Outputs:**
  - `processed/events` (partitioned by `year`)
  - `processed/spotify` (partitioned by `region`)
  - `processed/billboard`
  - `processed/audio`
- **Mini simulation output:** `Expanded event count: ~5,000,000`, `Spotify rows: ...`, `events rows: ...`.

### `03_feature_engineering.ipynb` (weekly features + label)

- **Purpose:** Build artist-week features and target label `charted`.
- **Inputs:** Parquet from notebook 2 + artist normalization across sources.
- **Outputs:** `processed/features` with columns such as:
  - momentum: `plays_1d`, `plays_7d`, `plays_28d`, `growth_rate_7d`
  - Spotify: `total_streams`, `region_spread`, `stream_velocity`
  - audio: `tempo`, `energy`, `loudness`, `danceability`
  - label: `charted` (Billboard hit in next 28 days)
- **Mini simulation output:** label distribution table like `charted=0/1` counts.

### `04_streaming.ipynb` (streaming simulation)

- **Purpose:** Simulate near-real-time momentum aggregation with Structured Streaming.
- **Inputs:** `processed/events`.
- **Outputs:**
  - streaming source files in `streaming/input`
  - append-mode windowed results in `streaming/output`
  - checkpoint state in `streaming/checkpoint`
- **Mini simulation output:** top artists by `play_count_7d` after short run (`time.sleep(120)`).

### `05_model_training.ipynb` (train + evaluate)

- **Purpose:** Train Random Forest and evaluate breakout prediction quality.
- **Inputs:** `processed/features`.
- **Outputs:** Printed metrics + saved model at `models/rf_model`.
- **Split logic:** Train on `week_year < 2021`; test on `week_year >= 2021`.
- **Mini simulation output:** `AUC-ROC: 0.8xxx`, `F1 Score: 0.xxxx`, confusion counts, feature importances.

### `06_dashboard.ipynb` (visual analysis)

- **Purpose:** Show model/business insights in 4 visual panels.
- **Inputs:** `processed/features`, `models/rf_model`, optional `streaming/output`.
- **Outputs:** Charts/widgets:
  1. latest-week momentum leaderboard
  2. artist trendline (dropdown)
  3. ROC + confusion matrix
  4. "novel positives" table (predicted hit but not labeled hit)
- **Mini simulation output:** interactive plots + `novel.show(20)` table.

### `07_project_report.ipynb` (final narrative)

- **Purpose:** Architecture summary, scalability rationale, and final evaluation notes.
- **Inputs:** Results from notebooks 1-6.
- **Outputs:** Human-readable report content for submission/presentation (not a data artifact).

### `Big Data Proposal.pdf` (project intent)

- **Purpose:** Original proposal: problem statement, intended pipeline, and success criteria.
- **Inputs:** Team planning assumptions.
- **Outputs:** Baseline scope to compare planned vs implemented system.

## Path contract (local-first)

All paths are rooted at:

`<repo>/data/`

Shared constants live in [`hdfs_paths.py`](hdfs_paths.py). Notebooks prepend the repo root to `sys.path` and import `hdfs_paths`.

Layout:

- `raw/` — Last.fm, Spotify charts, Billboard, MSD audio (CSV/TSV)
- `processed/` — Parquet: `events/`, `spotify/`, `billboard/`, `audio/`, `features/`
- `streaming/` — `input/`, `output/`, `checkpoint/`
- `models/` — `rf_model/` (saved MLlib pipeline)

## Spark settings

- `spark.sql.shuffle.partitions` defaults to **8** (local-friendly). Override with `BDCAP_SHUFFLE_PARTITIONS` for larger runs.
- Use **explicit CSV schemas** in ingestion cells.

## Datasets

Sources and manual download fallbacks are documented in notebook 01 (Kaggle may require auth).

## Requirements

PySpark 3.x environment (no HDFS CLI required). Notebook 06 may use `pip install scikit-learn --user` for ROC plotting only.
