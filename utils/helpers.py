"""
Utility helper functions for the application.
"""
import numpy as np
from typing import Union
from utils.logger import get_logger

logger = get_logger('helpers')


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """
    L2 normalize each row of a numpy array.

    Args:
        x: Input numpy array

    Returns:
        L2 normalized array
    """
    x = x.astype("float32", copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def sec_to_mmss(t: float) -> str:
    """
    Convert seconds to MM:SS format.

    Args:
        t: Time in seconds

    Returns:
        Time string in MM:SS format
    """
    if t < 0 or not np.isfinite(t):
        return "00:00"
    m = int(t // 60)
    s = int(round(t - m * 60))
    if s == 60:
        m, s = m + 1, 0
    return f"{m:02d}:{s:02d}"


def parse_mmss(s: str, default_sec: float = 0.0) -> float:
    """
    Parse 'MM:SS' or raw seconds string to float seconds.

    Args:
        s: Time string (MM:SS or seconds)
        default_sec: Default value to return on error

    Returns:
        Time in seconds
    """
    try:
        s = (s or "").strip()
        if not s:
            return float(default_sec)
        if ":" in s:
            parts = s.split(":", 1)
            mm = int(parts[0]) if parts[0] else 0
            ss = int(parts[1]) if parts[1] else 0
            return max(0.0, float(mm * 60 + ss))
        # fallback: raw seconds
        val = float(s)
        return max(0.0, val)
    except Exception as e:
        logger.warning(f"Error parsing time string '{s}': {e}")
        return float(default_sec)


def dataset_for_video(video_id: str) -> str:
    """
    Determine which dataset a video belongs to based on its ID.

    Args:
        video_id: Video identifier

    Returns:
        Dataset name ('b1' or 'b2')
    """
    return "b1" if video_id.startswith("L") else "b2"


def safe_divide(numerator: Union[int, float], denominator: Union[int, float], default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default value if denominator is zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if denominator is zero

    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default


def clamp(value: Union[int, float], min_value: Union[int, float], max_value: Union[int, float]) -> Union[int, float]:
    """
    Clamp a value between min and max values.

    Args:
        value: Value to clamp
        min_value: Minimum value
        max_value: Maximum value

    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))
