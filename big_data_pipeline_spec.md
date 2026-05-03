# Coding Agent Spec — Music Trend Detection Pipeline
## CS-GY 6513 Big Data | NYU Tandon | Spring 2026

---

## What You Are Building

A scalable big data pipeline that detects breakout songs from large-scale listening event data
before they appear on Billboard charts. The entire project is implemented as **six Jupyter
notebooks** that run sequentially on the course JupyterHub cluster with PySpark, HDFS, and
Spark Structured Streaming.

The pipeline has six stages that map directly to six notebooks:

```
NB1: Environment setup + data download
NB2: Batch ingestion → HDFS Parquet
NB3: Feature engineering (windowed aggregations, label join)
NB4: Spark Structured Streaming simulation
NB5: MLlib model training + evaluation
NB6: Interactive dashboard
```

---

## Constraints

- **Cluster environment**: NYU JupyterHub running PySpark 3.x on Hadoop/HDFS.
  No pip installs of heavy libraries — only what is already available in the cluster.
- **No Kafka required**: Kafka is described as the production ingestion layer but is
  NOT implemented. The streaming notebook uses Spark's built-in file-based streaming
  source to replay data chronologically, simulating what a Kafka consumer would do.
- **No MongoDB**: Storage is HDFS only. Do not introduce MongoDB.
- **No Dask**: Hyperparameter tuning uses Spark MLlib's CrossValidator, not Dask.
- **Single language**: All notebooks in Python (PySpark).
- **Scale target**: ~5 million rows of event data after synthetic expansion. This is
  the minimum size needed to force genuine distributed shuffle operations.

---

## Datasets

Download all datasets at the start of NB1. Store raw files in HDFS under `/user/<username>/music/raw/`.

### 1. Last.fm HetRec 2011
- **URL**: https://grouplens.org/datasets/hetrec-2011/
- **File**: `hetrec2011-lastfm-2k.zip` → extract `user_artists.dat`
- **Schema**: `userID \t artistID \t weight` (tab-separated, no timestamp)
- **Raw size**: ~92K rows (2,000 users × ~46 artists each)
- **What it represents**: Cumulative play counts per user-artist pair

> **Critical**: This dataset has no timestamps. You must synthetically expand it
> to ~5M rows with timestamps before it can be used as a streaming event log.

#### Synthetic Expansion Logic (implement in NB2)

```python
# For each (userID, artistID, weight) row, generate `weight` synthetic play events
# distributed across a 730-day window (2019-01-01 to 2020-12-31)
# Add controlled Gaussian noise to weights (σ = 0.1 × weight)
# Assign random timestamps within the window using numpy random

import numpy as np
from datetime import datetime, timedelta

BASE_DATE = datetime(2019, 1, 1)
WINDOW_DAYS = 730

def expand_row(user_id, artist_id, weight):
    # Number of events = weight (treat as play count)
    n_events = max(1, int(weight + np.random.normal(0, weight * 0.1)))
    # Random timestamps across the window
    offsets = np.random.randint(0, WINDOW_DAYS * 86400, size=n_events)
    timestamps = [BASE_DATE + timedelta(seconds=int(o)) for o in offsets]
    return [(user_id, artist_id, ts.strftime('%Y-%m-%d %H:%M:%S')) for ts in timestamps]
```

Document this expansion transparently in NB2 with a markdown cell explaining that
real statistical distributions are preserved.

---

### 2. Spotify Global Charts
- **URL**: https://www.kaggle.com/datasets/dhruvildave/spotify-charts
- **File**: `charts.csv`
- **Schema**: `title, rank, date, artist, url, region, chart, trend, streams`
- **Raw size**: ~5M rows
- **What it represents**: Weekly streaming counts by region, already at scale

---

### 3. Billboard Hot 100 (1958–2024)
- **URL**: https://www.kaggle.com/datasets/kcmillersean/billboard-hot-100
- **File**: `Hot 100.csv`
- **Schema**: `WeekID, Song, Performer, SongID, Instance, Previous Week Position, Peak Position, Weeks on Chart`
- **Raw size**: ~350K rows
- **Role**: Ground truth labels for the ML model

---

### 4. Million Song Dataset Audio Features
- **URL**: http://labrosa.ee.columbia.edu/millionsong/ → subset summary file
- **Alternative**: Download `msd_audio_features.csv` from the MSD challenge page
- **Schema**: `artist_name, song_title, tempo, energy, loudness, danceability, key, mode`
- **Raw size**: ~300MB
- **Role**: Static audio features joined to the event data on normalized artist name

---

## File / Directory Structure

```
/user/<username>/music/
├── raw/
│   ├── lastfm/          ← user_artists.dat
│   ├── spotify_charts/  ← charts.csv
│   ├── billboard/       ← Hot 100.csv
│   └── msd_audio/       ← msd_audio_features.csv
├── processed/
│   ├── events/          ← expanded Last.fm events (Parquet, partitioned by year)
│   ├── spotify/         ← Spotify charts (Parquet, partitioned by region)
│   ├── billboard/       ← Billboard (Parquet, no partition)
│   ├── audio/           ← MSD audio features (Parquet, no partition)
│   └── features/        ← Final artist-week feature table with labels (Parquet)
├── streaming/
│   ├── input/           ← Chunked event files for streaming simulation
│   └── output/          ← Streaming momentum scores (append mode)
└── models/
    └── rf_model/        ← Saved RandomForestClassificationModel
```

---

## Notebook 1 — Environment Verification & Data Download

**Filename**: `01_environment_and_data.ipynb`

### What it does

1. Verifies Spark session, HDFS connectivity, and available cores/memory
2. Downloads all four raw datasets from their URLs
3. Writes raw files to HDFS under `/user/<username>/music/raw/`
4. Prints a summary: row counts, file sizes, schema previews

### Key cells

```python
# Cell 1: Spark session
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("MusicTrend_01_Environment") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

sc = spark.sparkContext
print(f"Spark version: {spark.version}")
print(f"Default parallelism: {sc.defaultParallelism}")
```

```python
# Cell 2: HDFS check
import subprocess

result = subprocess.run(
    ["hdfs", "dfs", "-ls", "/user"],
    capture_output=True, text=True
)
print(result.stdout)
```

```python
# Cell 3: Download datasets
# Use wget or requests to download each file
# Upload to HDFS using: hdfs dfs -put local_file hdfs_path
```

```python
# Cell 4: Schema preview for each dataset
lastfm_df = spark.read.option("sep", "\t").option("header", True).csv("hdfs:///user/.../raw/lastfm/user_artists.dat")
lastfm_df.printSchema()
lastfm_df.show(5)
print(f"Last.fm row count: {lastfm_df.count()}")
```

---

## Notebook 2 — Batch Ingestion & Parquet Storage

**Filename**: `02_ingestion_parquet.ipynb`

### What it does

1. Reads each raw dataset from HDFS
2. Cleans and type-casts all schemas explicitly
3. Runs the synthetic expansion on Last.fm data → 5M+ timestamped events
4. Writes all four datasets to HDFS as partitioned Parquet
5. Verifies partition counts and row totals after write

### Schemas after cleaning

**Events (expanded Last.fm)**:
```
user_id: string
artist_id: string
event_timestamp: timestamp
```
Partitioned by: `year` (derived from `event_timestamp`)

**Spotify Charts**:
```
title: string
rank: integer
chart_date: date
artist: string
region: string
streams: long
```
Partitioned by: `region`

**Billboard Hot 100**:
```
week_date: date
song: string
performer: string
song_id: string
peak_position: integer
weeks_on_chart: integer
```
No partition.

**MSD Audio Features**:
```
artist_name: string
tempo: double
energy: double
loudness: double
danceability: double
```
No partition.

### Synthetic Expansion (implement here)

```python
from pyspark.sql import Row
from pyspark.sql.functions import col, year

# Read raw Last.fm
raw = spark.read.option("sep","\t").option("header",True) \
    .csv("hdfs:///user/.../raw/lastfm/user_artists.dat")

# Expand using mapPartitions for efficiency
def expand_partition(rows):
    import numpy as np
    from datetime import datetime, timedelta
    BASE = datetime(2019, 1, 1)
    WINDOW = 730 * 86400
    for row in rows:
        weight = max(1, int(float(row.weight)))
        n = max(1, int(weight + np.random.normal(0, weight * 0.1)))
        offsets = np.random.randint(0, WINDOW, size=n)
        for o in offsets:
            ts = BASE + timedelta(seconds=int(o))
            yield Row(user_id=row.userID,
                      artist_id=row.artistID,
                      event_timestamp=ts.strftime('%Y-%m-%d %H:%M:%S'))

expanded_rdd = raw.rdd.mapPartitions(expand_partition)
events_df = spark.createDataFrame(expanded_rdd) \
    .withColumn("event_timestamp", col("event_timestamp").cast("timestamp")) \
    .withColumn("year", year("event_timestamp"))

print(f"Expanded event count: {events_df.count()}")  # Should be ~5M

# Write partitioned by year
events_df.write.mode("overwrite") \
    .partitionBy("year") \
    .parquet("hdfs:///user/.../processed/events/")
```

Add a markdown cell explaining the expansion methodology and confirming it preserves
the original statistical distribution of listening behavior.

---

## Notebook 3 — Feature Engineering & Label Generation

**Filename**: `03_feature_engineering.ipynb`

### What it does

1. Reads the processed events and Spotify Charts Parquet from HDFS
2. Normalizes artist names across both datasets (lowercase, strip punctuation)
3. Computes per-artist, per-week features using Spark window functions
4. Joins with Billboard data to generate binary classification labels
5. Writes the final labeled feature table to HDFS

### Artist Name Normalization

```python
from pyspark.sql.functions import lower, regexp_replace, trim

def normalize_artist(col_name):
    return trim(lower(regexp_replace(col_name, r"[^a-z0-9\s]", "")))

events_df = events_df.withColumn("artist_norm", normalize_artist(col("artist_id")))
spotify_df = spotify_df.withColumn("artist_norm", normalize_artist(col("artist")))
billboard_df = billboard_df.withColumn("artist_norm", normalize_artist(col("performer")))
```

### Feature Computation (window functions)

Compute these features **per artist per ISO week**:

| Feature | Description | How to compute |
|---|---|---|
| `plays_1d` | Play count in last 1 day | Window: current day, partitioned by artist |
| `plays_7d` | Play count in last 7 days | Window: 7-day rolling, partitioned by artist |
| `plays_28d` | Play count in last 28 days | Window: 28-day rolling, partitioned by artist |
| `growth_rate_7d` | plays_7d / plays_28d | Ratio, higher = accelerating |
| `stream_velocity` | Week-over-week Δ in avg streams | Join with Spotify by artist+week |
| `region_spread` | Count of distinct regions with rising Spotify streams | Group by artist+week on Spotify |

```python
from pyspark.sql import Window
from pyspark.sql.functions import (
    col, sum as _sum, count, countDistinct,
    date_trunc, weekofyear, year, to_date, lag
)

# Aggregate events to artist-day grain
events_daily = events_df \
    .withColumn("event_date", to_date("event_timestamp")) \
    .groupBy("artist_norm", "event_date") \
    .agg(count("*").alias("daily_plays"))

# Define rolling windows
W7  = Window.partitionBy("artist_norm") \
    .orderBy(col("event_date").cast("long")) \
    .rangeBetween(-6 * 86400, 0)  # 7-day window in seconds

W28 = Window.partitionBy("artist_norm") \
    .orderBy(col("event_date").cast("long")) \
    .rangeBetween(-27 * 86400, 0)  # 28-day window

features_df = events_daily \
    .withColumn("plays_7d",  _sum("daily_plays").over(W7)) \
    .withColumn("plays_28d", _sum("daily_plays").over(W28)) \
    .withColumn("growth_rate_7d", col("plays_7d") / (col("plays_28d") + 1))

# Aggregate to weekly grain for label join
from pyspark.sql.functions import max as _max

weekly = features_df \
    .withColumn("week_year",   year("event_date")) \
    .withColumn("week_number", weekofyear("event_date")) \
    .groupBy("artist_norm", "week_year", "week_number") \
    .agg(
        _max("plays_7d").alias("plays_7d"),
        _max("plays_28d").alias("plays_28d"),
        _max("growth_rate_7d").alias("growth_rate_7d")
    )
```

### Spotify Join (stream_velocity and region_spread)

```python
# Compute week-over-week stream velocity per artist per week in Spotify data
W_lag = Window.partitionBy("artist_norm").orderBy("chart_date")

spotify_weekly = spotify_df \
    .withColumn("week_year",   year("chart_date")) \
    .withColumn("week_number", weekofyear("chart_date")) \
    .groupBy("artist_norm", "week_year", "week_number") \
    .agg(
        _sum("streams").alias("total_streams"),
        countDistinct("region").alias("region_spread")
    )

spotify_with_velocity = spotify_weekly \
    .withColumn("prev_streams", lag("total_streams", 1).over(
        Window.partitionBy("artist_norm").orderBy("week_year", "week_number")
    )) \
    .withColumn("stream_velocity",
        (col("total_streams") - col("prev_streams")) / (col("prev_streams") + 1)
    )

# Join events features with Spotify features
combined = weekly.join(
    spotify_with_velocity,
    on=["artist_norm", "week_year", "week_number"],
    how="left"
).fillna(0, subset=["total_streams", "region_spread", "stream_velocity"])
```

### Billboard Label Generation

For each artist-week snapshot, label it `1` if the artist appears on Billboard
**within the following 4 weeks**, else `0`.

```python
from pyspark.sql.functions import broadcast

# Get all (artist, week) pairs where the artist was on Billboard
billboard_weeks = billboard_df \
    .withColumn("week_year",   year("week_date")) \
    .withColumn("week_number", weekofyear("week_date")) \
    .select("artist_norm", "week_year", "week_number") \
    .distinct()

# For each feature row, check if artist charted in any of the next 4 weeks
# Approach: cross-join with a small offset table, then anti/semi join
from pyspark.sql.functions import lit, array, explode

offsets_df = spark.createDataFrame(
    [(1,), (2,), (3,), (4,)], ["offset_weeks"]
)

future_chart = combined \
    .crossJoin(broadcast(offsets_df)) \
    .withColumn("future_week", col("week_number") + col("offset_weeks")) \
    .join(billboard_weeks,
          on=["artist_norm", "week_year"],
          how="left_semi") \
    .select("artist_norm", "week_year", "week_number") \
    .distinct() \
    .withColumn("charted", lit(1))

labeled = combined.join(future_chart,
    on=["artist_norm", "week_year", "week_number"], how="left"
).fillna(0, subset=["charted"])

print(f"Label distribution:\n{labeled.groupBy('charted').count().show()}")
```

### Write Feature Store

```python
labeled.write.mode("overwrite") \
    .parquet("hdfs:///user/.../processed/features/")
```

---

## Notebook 4 — Spark Structured Streaming Simulation

**Filename**: `04_streaming.ipynb`

### What it does

1. Chunks the expanded event Parquet into time-ordered JSON files
2. Uses `spark.readStream` with a file-based source to simulate live event replay
3. Applies a watermark + sliding window to compute rolling momentum scores
4. Writes live scores to HDFS in append mode

### Why this is a valid real-time demo

In production, this job would read from a Kafka topic partitioned by `artist_id`.
The `spark.readStream` API is **identical** whether the source is a file directory
or a Kafka broker — only the source configuration changes. All watermarking,
windowing, and state management logic is unchanged.

### Step 1: Chunk events into time-ordered input files

```python
# Read expanded events, sort by timestamp, split into hourly chunks
events = spark.read.parquet("hdfs:///user/.../processed/events/")

from pyspark.sql.functions import date_trunc

# Write to streaming input directory as individual JSON files per hour-chunk
events \
    .withColumn("hour_chunk", date_trunc("hour", "event_timestamp")) \
    .write.mode("overwrite") \
    .partitionBy("hour_chunk") \
    .json("hdfs:///user/.../streaming/input/")
```

### Step 2: Define streaming query

```python
from pyspark.sql.types import StructType, StringType, TimestampType
from pyspark.sql.functions import window, count, col

# Schema for incoming events
event_schema = StructType() \
    .add("user_id", StringType()) \
    .add("artist_id", StringType()) \
    .add("event_timestamp", TimestampType())

# Read as a stream from the file directory
streaming_events = spark.readStream \
    .schema(event_schema) \
    .option("maxFilesPerTrigger", 1) \
    .json("hdfs:///user/.../streaming/input/")

# Apply watermark and sliding window aggregation
momentum_stream = streaming_events \
    .withWatermark("event_timestamp", "1 hour") \
    .groupBy(
        window(col("event_timestamp"), "7 days", "1 day"),  # 7-day window, 1-day slide
        col("artist_id")
    ) \
    .agg(count("*").alias("play_count_7d"))

# Write to output in append mode
query = momentum_stream.writeStream \
    .outputMode("append") \
    .format("json") \
    .option("path", "hdfs:///user/.../streaming/output/") \
    .option("checkpointLocation", "hdfs:///user/.../streaming/checkpoint/") \
    .trigger(processingTime="30 seconds") \
    .start()

# Run for 5 minutes to generate output, then stop
import time
time.sleep(300)
query.stop()
```

### Step 3: Verify output

```python
# Read streaming output and show top momentum artists
output = spark.read.json("hdfs:///user/.../streaming/output/")
output.orderBy(col("play_count_7d").desc()).show(20)
```

Add a markdown cell explaining what a production Kafka configuration would look like
(just the readStream config block, not actual Kafka setup).

---

## Notebook 5 — MLlib Model Training & Evaluation

**Filename**: `05_model_training.ipynb`

### What it does

1. Reads the labeled feature store from HDFS
2. Performs a **time-aware** train/test split (not random)
3. Builds a Spark MLlib pipeline: VectorAssembler → StandardScaler → RandomForestClassifier
4. Trains the model and evaluates with AUC-ROC, F1, confusion matrix
5. Extracts feature importances
6. Saves the trained model to HDFS

### Feature columns

```python
FEATURE_COLS = [
    "plays_7d",
    "plays_28d",
    "growth_rate_7d",
    "stream_velocity",
    "region_spread",
    "total_streams"
]
LABEL_COL = "charted"
```

Fill nulls with 0 before assembling features.

### Time-aware train/test split

```python
# CRITICAL: Do NOT use randomSplit() — that leaks future labels into training
# Train on data before 2021, test on 2021 onward
# This prevents the model from learning from chart outcomes it wouldn't have
# access to in a real deployment scenario

features = spark.read.parquet("hdfs:///user/.../processed/features/") \
    .fillna(0, subset=FEATURE_COLS)

train = features.filter(col("week_year") < 2021)
test  = features.filter(col("week_year") >= 2021)

print(f"Train size: {train.count()}, Test size: {test.count()}")
print(f"Train label dist: {train.groupBy('charted').count().show()}")
print(f"Test label dist:  {test.groupBy('charted').count().show()}")
```

### MLlib Pipeline

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

assembler = VectorAssembler(
    inputCols=FEATURE_COLS,
    outputCol="raw_features",
    handleInvalid="keep"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True, withMean=True
)

rf = RandomForestClassifier(
    labelCol=LABEL_COL,
    featuresCol="features",
    numTrees=100,
    maxDepth=8,
    seed=42
)

pipeline = Pipeline(stages=[assembler, scaler, rf])
model = pipeline.fit(train)
```

### Evaluation

```python
predictions = model.transform(test)

# AUC-ROC
auc_evaluator = BinaryClassificationEvaluator(
    labelCol=LABEL_COL, metricName="areaUnderROC"
)
auc = auc_evaluator.evaluate(predictions)
print(f"AUC-ROC: {auc:.4f}")

# F1
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol=LABEL_COL, metricName="f1"
)
f1 = f1_evaluator.evaluate(predictions)
print(f"F1 Score: {f1:.4f}")

# Confusion matrix
predictions.groupBy(LABEL_COL, "prediction").count().show()
```

### Feature importances

```python
rf_model = model.stages[-1]
importances = list(zip(FEATURE_COLS, rf_model.featureImportances.toArray()))
importances.sort(key=lambda x: x[1], reverse=True)
print("Feature importances:")
for feat, imp in importances:
    print(f"  {feat}: {imp:.4f}")
```

### Save model

```python
model.write().overwrite().save("hdfs:///user/.../models/rf_model/")
```

---

## Notebook 6 — Interactive Dashboard

**Filename**: `06_dashboard.ipynb`

### What it does

1. Reads the feature store + streaming output + model predictions from HDFS
2. Renders four interactive panels using `matplotlib` and `ipywidgets`

### Panel 1: Breakout candidate leaderboard

Rank artists by their peak `growth_rate_7d` score in the most recent week of data.
Show top 20 with a horizontal bar chart. Color bars by whether the artist eventually
charted (green) or didn't (gray).

### Panel 2: Per-artist momentum trendline

`ipywidgets.Dropdown` populated with the top 50 artists by momentum. On selection,
plot a line chart of `plays_7d` over time for that artist, with a vertical dashed
line at the week they first appeared on Billboard (if they did).

```python
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

artist_list = leaderboard_df.toPandas()["artist_norm"].tolist()
dropdown = widgets.Dropdown(options=artist_list, description="Artist:")

def plot_artist(artist):
    data = features_pd[features_pd["artist_norm"] == artist].sort_values("week_number")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data["week_number"], data["plays_7d"], label="7-day plays")
    # Mark chart entry if exists
    chart_week = data[data["charted"] == 1]["week_number"].min()
    if not pd.isna(chart_week):
        ax.axvline(x=chart_week, color="red", linestyle="--", label="Chart entry")
    ax.set_title(f"Momentum: {artist}")
    ax.legend()
    plt.show()

widgets.interactive(plot_artist, artist=dropdown)
```

### Panel 3: Model evaluation

Show ROC curve (using sklearn on the Pandas-converted prediction output) and
a formatted confusion matrix heatmap using `matplotlib.pyplot.imshow`.

```python
from sklearn.metrics import roc_curve, auc as sk_auc
import numpy as np

preds_pd = predictions.select("charted", "probability", "prediction").toPandas()
preds_pd["prob_1"] = preds_pd["probability"].apply(lambda x: float(x[1]))

fpr, tpr, _ = roc_curve(preds_pd["charted"], preds_pd["prob_1"])
roc_auc = sk_auc(fpr, tpr)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC curve
axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
axes[0].plot([0,1],[0,1], "k--")
axes[0].set_title("ROC Curve")
axes[0].legend()

# Confusion matrix
cm = predictions.groupBy("charted","prediction").count().toPandas()
# ... render as heatmap
axes[1].set_title("Confusion Matrix")

plt.tight_layout()
plt.show()
```

### Panel 4: Novel predictions

Show artists that the model predicts will chart (`prediction == 1`) but have NOT
yet appeared on Billboard in the test set (`charted == 0`). These are the
"before they chart" candidates — the core deliverable of the project.

```python
novel = predictions.filter(
    (col("prediction") == 1) & (col(LABEL_COL) == 0)
).select("artist_norm", "week_year", "week_number", "probability", "growth_rate_7d") \
 .orderBy(col("probability").desc())

novel.show(20, truncate=False)
```

---

## Scalability Argument (for report)

Three concrete bottlenecks require Spark's distributed execution:

1. **Shuffle cost of multi-window aggregations**: Computing simultaneous 1-day, 7-day,
   and 28-day rolling windows over 5M timestamped rows partitioned by artist requires
   a multi-stage shuffle where data is redistributed across workers by artist ID,
   aggregated locally, then merged. The intermediate shuffle size exceeds single-machine RAM.

2. **Cross-dataset join**: Joining 5M Last.fm events with 5M Spotify rows on normalized
   artist name requires a shuffle join. Spark's broadcast join optimization applies for
   the smaller Billboard dataset (~350K rows), demonstrably reducing network transfer.

3. **Stateful streaming aggregation**: The watermarked sliding window in NB4 requires
   stateful aggregation with automatic backpressure — a property single-machine
   processing cannot provide at production event rates.

---

## Deliverables Checklist

- [ ] `01_environment_and_data.ipynb` — runs clean, all datasets verified in HDFS
- [ ] `02_ingestion_parquet.ipynb` — 5M+ events confirmed, Parquet partitions written
- [ ] `03_feature_engineering.ipynb` — feature table with label distribution printed
- [ ] `04_streaming.ipynb` — streaming query runs, output files in HDFS
- [ ] `05_model_training.ipynb` — AUC-ROC and F1 printed, model saved
- [ ] `06_dashboard.ipynb` — all 4 panels render, dropdown works
- [ ] Final report notebook (can be appended to NB6) covering architecture decisions,
      observed shuffle metrics, and model evaluation

---

## Notes for the Agent

- When a Kaggle dataset URL is unreachable, download the CSV manually and use
  `hdfs dfs -put` to upload it; add a markdown cell explaining this in NB1.
- If the JupyterHub cluster lacks `sklearn`, install it with
  `!pip install scikit-learn --user` in NB6 (only used for the ROC curve, not for model training).
- All Spark DataFrames must use explicit schemas — never infer from CSV in production cells.
- Use `spark.sql.shuffle.partitions=50` for the cluster size; do not leave it at default 200.
- Each notebook must start with a markdown cell describing what it does and end with
  a markdown cell confirming its outputs.
- Notebooks must be runnable independently after NB2 (each reads from HDFS, not from
  the previous notebook's variables).
