"""
Video service for managing video files, keyframes, and metadata.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image
from utils.logger import get_logger
from utils.helpers import dataset_for_video
from config.settings import B1, B2

logger = get_logger('video_service')


def keyframe_path(video_id: str, frame_id: int) -> str:
    """
    Get the path to a keyframe image.

    Args:
        video_id: Video identifier
        frame_id: Frame number

    Returns:
        Path to the keyframe image file
    """
    D = B1 if dataset_for_video(video_id) == "b1" else B2
    L, V = video_id.split("_")
    base = f"{D['kf_root']}/{L}/{V}"
    digits = (3, 4) if dataset_for_video(video_id) == "b1" else (4, 3)
    frames_try = (frame_id, max(0, frame_id - 1))
    exts = (".jpg", ".jpeg", ".png")

    for d in digits:
        for fr in frames_try:
            for ext in exts:
                p = f"{base}/{fr:0{d}d}{ext}"
                if os.path.exists(p):
                    return p

    return f"{base}/{frame_id:0{digits[0]}d}.jpg"


def npy_path_for_video(video_id: str) -> str:
    """
    Get the path to CLIP features numpy file for a video.

    Args:
        video_id: Video identifier

    Returns:
        Path to the .npy file
    """
    D = B1 if dataset_for_video(video_id) == "b1" else B2
    return f"{D['clip_dir']}/{video_id}.npy"


def map_csv_path_for_video(video_id: str) -> str:
    """
    Get the path to the map-keyframes CSV file for a video.

    Args:
        video_id: Video identifier

    Returns:
        Path to the CSV file
    """
    D = B1 if dataset_for_video(video_id) == "b1" else B2
    return f"{D['map_csv']}/{video_id}.csv"


def keep_idxs_path_for_video(video_id: str) -> str:
    """
    Get the path to the keep_idxs numpy file for a video.

    Args:
        video_id: Video identifier

    Returns:
        Path to the keep_idxs .npy file
    """
    D = B1 if dataset_for_video(video_id) == "b1" else B2
    return f"{D['keep_dir']}/{video_id}_keep_idxs.npy"


def safe_imread(path: str) -> Optional[Image.Image]:
    """
    Safely read an image file.

    Args:
        path: Path to the image file

    Returns:
        PIL Image object or None if file doesn't exist or error occurs
    """
    try:
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
        else:
            logger.warning(f"Image file not found: {path}")
            return None
    except Exception as e:
        logger.error(f"Error reading image {path}: {e}")
        return None


def get_fps_from_csv(video_id: str) -> Optional[float]:
    """
    Get FPS from the map-keyframes CSV file.

    Args:
        video_id: Video identifier

    Returns:
        FPS value or None if not found
    """
    try:
        csv_path = map_csv_path_for_video(video_id)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'fps' in df.columns and not df.empty:
                fps = float(df['fps'].iloc[0])
                logger.debug(f"FPS for {video_id}: {fps}")
                return fps
    except Exception as e:
        logger.error(f"Error getting FPS for {video_id}: {e}")

    # Default FPS
    return 25.0


def timestamp_to_keyframe_number(video_id: str, timestamp: float) -> int:
    """
    Convert timestamp to keyframe number using map-keyframes CSV.

    Args:
        video_id: Video identifier
        timestamp: Timestamp in seconds

    Returns:
        Keyframe number
    """
    try:
        # Try to read map-keyframes CSV for exact mapping
        csv_path = map_csv_path_for_video(video_id)

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            if 'pts_time' in df.columns and 'n' in df.columns:
                # Find the keyframe with timestamp closest to the target
                df['time_diff'] = abs(df['pts_time'] - timestamp)
                closest_row = df.loc[df['time_diff'].idxmin()]

                # Return the keyframe number (n column)
                keyframe_num = int(closest_row['n'])
                logger.debug(f"Found exact mapping: timestamp {timestamp:.2f}s -> keyframe {keyframe_num}")
                return keyframe_num

    except Exception as e:
        logger.warning(f"Error reading map CSV for {video_id}: {e}")

    # Fallback: estimate based on typical keyframe extraction rate
    fps = get_fps_from_csv(video_id) or 25.0
    estimated_frame = int(timestamp * fps)

    # For batch 2, keyframes seem to be more frequent
    if dataset_for_video(video_id) == "b1":
        keyframe_interval = 25  # ~1 second at 25fps
    else:  # batch 2
        keyframe_interval = 50  # More conservative for batch 2

    estimated_keyframe = max(1, estimated_frame // keyframe_interval)
    logger.debug(f"Using estimation: timestamp {timestamp:.2f}s -> estimated keyframe {estimated_keyframe}")

    return estimated_keyframe


def load_times_for_video(video_id: str, n_feats: int) -> np.ndarray:
    """
    Load timestamps for each keyframe in a video.

    Args:
        video_id: Video identifier
        n_feats: Number of features/frames

    Returns:
        Array of timestamps
    """
    try:
        csv_path = map_csv_path_for_video(video_id)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'pts_time' in df.columns:
                times = df['pts_time'].values[:n_feats]
                if len(times) < n_feats:
                    # Pad with estimates if needed
                    fps = get_fps_from_csv(video_id) or 25.0
                    missing = n_feats - len(times)
                    last_time = times[-1] if len(times) > 0 else 0.0
                    padding = np.arange(1, missing + 1) * (1.0 / fps) + last_time
                    times = np.concatenate([times, padding])
                logger.debug(f"Loaded {len(times)} timestamps for {video_id}")
                return times
    except Exception as e:
        logger.warning(f"Error loading times for {video_id}: {e}")

    # Fallback: generate uniform timestamps
    fps = get_fps_from_csv(video_id) or 25.0
    times = np.arange(n_feats) / fps
    logger.debug(f"Generated {len(times)} estimated timestamps for {video_id}")
    return times


def load_video_features(video_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load CLIP features and timestamps for a video.

    Args:
        video_id: Video identifier

    Returns:
        Tuple of (features, timestamps) or None if not found
    """
    try:
        npy_file = npy_path_for_video(video_id)
        if not os.path.exists(npy_file):
            logger.warning(f"Features file not found: {npy_file}")
            return None

        features = np.load(npy_file).astype("float32")

        # Try to load keep_idxs to filter features
        keep_idxs_file = keep_idxs_path_for_video(video_id)
        if os.path.exists(keep_idxs_file):
            keep_idxs = np.load(keep_idxs_file)
            features = features[keep_idxs]
            logger.debug(f"Filtered features using keep_idxs: {len(keep_idxs)} frames")

        # Load timestamps
        timestamps = load_times_for_video(video_id, len(features))

        logger.info(f"Loaded features for {video_id}: shape={features.shape}")
        return features, timestamps

    except Exception as e:
        logger.error(f"Error loading features for {video_id}: {e}")
        return None


def list_videos_from_dirs(clip_dirs: List[str], limit_videos: Optional[int] = None,
                         exclude_ser: Optional[set] = None) -> List[str]:
    """
    List all videos from CLIP feature directories.

    Args:
        clip_dirs: List of directories containing CLIP features
        limit_videos: Maximum number of videos to return
        exclude_ser: Set of video series to exclude

    Returns:
        List of video IDs
    """
    if exclude_ser is None:
        exclude_ser = set()

    vids = []
    for clip_dir in clip_dirs:
        if not os.path.isdir(clip_dir):
            continue
        for fname in os.listdir(clip_dir):
            if fname.endswith(".npy"):
                vid = fname[:-4]  # Remove .npy extension
                ser = vid.split("_")[0] if "_" in vid else ""
                if ser not in exclude_ser:
                    vids.append(vid)

    logger.info(f"Found {len(vids)} videos in {len(clip_dirs)} directories")

    if limit_videos is not None and limit_videos > 0:
        vids = vids[:limit_videos]
        logger.info(f"Limited to {len(vids)} videos")

    return vids


def get_video_info(video_id: str) -> Optional[dict]:
    """
    Get metadata information for a video.

    Args:
        video_id: Video identifier

    Returns:
        Dictionary with video metadata or None if not found
    """
    try:
        csv_path = map_csv_path_for_video(video_id)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            info = {
                'video_id': video_id,
                'dataset': dataset_for_video(video_id),
                'num_frames': len(df),
                'fps': get_fps_from_csv(video_id),
                'duration': df['pts_time'].max() if 'pts_time' in df.columns else None
            }
            return info
    except Exception as e:
        logger.error(f"Error getting video info for {video_id}: {e}")

    return None
