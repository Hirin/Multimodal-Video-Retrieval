"""
Configuration settings for the Multimodal Video Retrieval application.
"""
import os
import glob
from pathlib import Path
from typing import Optional, Set


def load_env_file(env_path: str = '.env') -> None:
    """
    Load environment variables from .env file.

    Args:
        env_path: Path to the .env file
    """
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")


def find_db_path(data_root: str, batch_name: str) -> Optional[str]:
    """
    Find the BM25 database path for a given batch.

    Args:
        data_root: Root directory for data
        batch_name: Name of the batch (e.g., 'b1', 'b2')

    Returns:
        Path to the database file or None if not found
    """
    # Try common locations
    possible_paths = [
        f"{data_root}{batch_name}/text_bm25.db",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # Search recursively for .db files
    search_pattern = f"{data_root}{batch_name}/**/*.db"
    db_files = glob.glob(search_pattern, recursive=True)
    bm25_files = [f for f in db_files if 'bm25' in f.lower() or 'text' in f.lower()]

    if bm25_files:
        return bm25_files[0]

    return None


# Load environment variables
load_env_file()

# Data root directory
DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/shared/HCMC-AI/")

# API Keys
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_API_KEY = GOOGLE_API_KEY

# Search Configuration
DEFAULT_TOP_K = 20
TOP_VIDEOS = 20
LIMIT_VIDEOS = None
EXCLUDE_SER: Set[str] = set()

# Batch 1 Configuration
B1 = {
    "clip_dir": f"{DATA_ROOT}b1/clip-features-32",
    "map_csv": f"{DATA_ROOT}b1/map-keyframes",
    "kf_root": f"{DATA_ROOT}b1/keyframes",
    "vid_root": f"{DATA_ROOT}b1/video",
    "keep_dir": f"{DATA_ROOT}b1/clip_keep_idxs",
}

# Batch 2 Configuration
B2 = {
    "clip_dir": f"{DATA_ROOT}b2/clip-features-32",
    "map_csv": f"{DATA_ROOT}b2/map-keyframes",
    "kf_root": f"{DATA_ROOT}b2/keyframes",
    "vid_root": f"{DATA_ROOT}b2/video",
    "keep_dir": f"{DATA_ROOT}b2/clip_keep_idxs",
}

# Database paths
B1_DB_PATH = find_db_path(DATA_ROOT, "b1")
B2_DB_PATH = find_db_path(DATA_ROOT, "b2")

# Available datasets
DATASETS = {"b1", "b2"}

# Gemini Configuration
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MAX_TOKENS = 10000
USE_LLM_ASPECTS = bool(GEMINI_API_KEY)

# Video Processing Configuration
WINDOW_SEC = 40.0
SKIP_SEC = 0.0
DELTA_SEC = 20.0

# Local Video Configuration
LOCAL_VIDEO_ROOT = Path("./videos")
LOCAL_VIDEO_URL_BASE = "http://127.0.0.1:8801/video"

# Server Configuration
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 8801


def get_batch_config(batch_name: str) -> dict:
    """
    Get configuration for a specific batch.

    Args:
        batch_name: Name of the batch ('b1' or 'b2')

    Returns:
        Configuration dictionary for the batch

    Raises:
        ValueError: If batch_name is not valid
    """
    if batch_name == "b1":
        return B1
    elif batch_name == "b2":
        return B2
    else:
        raise ValueError(f"Invalid batch name: {batch_name}. Must be 'b1' or 'b2'")


def get_db_path(batch_name: str) -> Optional[str]:
    """
    Get database path for a specific batch.

    Args:
        batch_name: Name of the batch ('b1' or 'b2')

    Returns:
        Path to the database file or None if not found
    """
    if batch_name == "b1":
        return B1_DB_PATH
    elif batch_name == "b2":
        return B2_DB_PATH
    else:
        return None
