"""
Shared HDFS layout for the Music Trend pipeline (NYU JupyterHub).
Import from notebooks: sys.path.insert(0, os.path.abspath(".")); import hdfs_paths
"""
import os

USER = os.environ.get("USER", "hdfs")
BASE = f"hdfs:///user/{USER}/music"

RAW_LASTFM = f"{BASE}/raw/lastfm/user_artists.dat"
RAW_SPOTIFY = f"{BASE}/raw/spotify_charts/charts.csv"
RAW_BILLBOARD = f"{BASE}/raw/billboard/Hot 100.csv"
RAW_MSD = f"{BASE}/raw/msd_audio/msd_audio_features.csv"

PROCESSED_EVENTS = f"{BASE}/processed/events"
PROCESSED_SPOTIFY = f"{BASE}/processed/spotify"
PROCESSED_BILLBOARD = f"{BASE}/processed/billboard"
PROCESSED_AUDIO = f"{BASE}/processed/audio"
PROCESSED_FEATURES = f"{BASE}/processed/features"

STREAMING_INPUT = f"{BASE}/streaming/input"
STREAMING_OUTPUT = f"{BASE}/streaming/output"
STREAMING_CHECKPOINT = f"{BASE}/streaming/checkpoint"

MODEL_RF = f"{BASE}/models/rf_model"

# Spark tuning (course cluster)
SHUFFLE_PARTITIONS = "50"
