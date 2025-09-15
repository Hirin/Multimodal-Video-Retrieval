import os
import io
import re
import json
import glob
import time
import sqlite3
import concurrent.futures
from pathlib import Path
from collections import deque
from typing import List, Dict, Tuple, Optional

import shutil
import threading
from flask import Flask, request, Response, send_file

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import streamlit.components.v1 as components
from deep_translator import GoogleTranslator

# Reset watch mode keys on app start/refresh to default to Youtube Online
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    # Clear all watch mode keys to default to Youtube Online
    keys_to_reset = [key for key in st.session_state.keys() if key.startswith("watch_mode_")]
    for key in keys_to_reset:
        del st.session_state[key]

# ============================ KAGGLE DOWNLOADER ============================
def create_popover_css():
    """Create CSS to improve popover size and appearance"""
    return """
    <style>
    .stPopover {
        min-width: 450px !important;
        min-height: 100px !important;
        padding: 25px !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15) !important;
    }
    .stPopover > div {
        padding: 25px !important;
        min-height: 80px !important;
    }
    .stPopover [data-testid="stRadio"] {
        margin: 10px 0 !important;
    }
    .stPopover [data-testid="stRadio"] > div {
        gap: 15px !important;
    }
    </style>
    """

def create_loading_overlay_html(message: str):
    """Create a custom loading overlay as HTML component"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            html, body {{
                width: 100%;
                height: 100%;
                overflow: hidden;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            @keyframes pulse {{
                0%, 100% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
            }}
            @keyframes progress {{
                0% {{ width: 0%; }}
                50% {{ width: 70%; }}
                100% {{ width: 100%; }}
            }}
            .loading-overlay {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: rgba(0, 0, 0, 0.85);
                backdrop-filter: blur(10px);
                z-index: 999999;
                display: flex;
                justify-content: center;
                align-items: center;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            .loading-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 80px 100px;
                border-radius: 30px;
                box-shadow: 0 30px 60px rgba(0, 0, 0, 0.5);
                text-align: center;
                color: white;
                min-width: 600px;
                max-width: 700px;
                width: 90%;
                animation: pulse 2s infinite;
            }}
            .loading-icon {{
                font-size: 80px;
                margin-bottom: 40px;
                animation: spin 1s linear infinite;
            }}
            .loading-title {{
                margin: 0 0 20px 0;
                font-size: 36px;
                font-weight: 700;
            }}
            .loading-message {{
                margin: 0;
                font-size: 20px;
                opacity: 0.9;
                line-height: 1.6;
            }}
            .progress-bar {{
                margin-top: 40px;
                width: 100%;
                height: 8px;
                background: rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                overflow: hidden;
            }}
            .progress-fill {{
                height: 100%;
                background: linear-gradient(90deg, #00d4ff, #00ff88);
                border-radius: 4px;
                animation: progress 2s ease-in-out infinite;
            }}
            .loading-subtitle {{
                margin-top: 25px;
                font-size: 16px;
                opacity: 0.7;
            }}
        </style>
    </head>
    <body>
        <div class="loading-overlay">
            <div class="loading-card">
                <div class="loading-icon">⏳</div>
                <h2 class="loading-title">Đang tải video...</h2>
                <p class="loading-message">{message}</p>
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
                <div class="loading-subtitle">Vui lòng đợi trong giây lát...</div>
            </div>
        </div>
    </body>
    </html>
    """

def create_loading_overlay(message: str):
    """Create a custom loading overlay with blur background"""
    return f"""
    <div style="
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(5px);
        z-index: 9999;
        display: flex;
        justify-content: center;
        align-items: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    ">
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            text-align: center;
            color: white;
            max-width: 400px;
            animation: pulse 2s infinite;
        ">
            <div style="
                font-size: 48px;
                margin-bottom: 20px;
                animation: spin 1s linear infinite;
            ">⏳</div>
            <h3 style="margin: 0 0 10px 0; font-size: 24px; font-weight: 600;">Đang tải video...</h3>
            <p style="margin: 0; font-size: 16px; opacity: 0.9;">{message}</p>
            <div style="
                margin-top: 20px;
                width: 100%;
                height: 4px;
                background: rgba(255, 255, 255, 0.3);
                border-radius: 2px;
                overflow: hidden;
            ">
                <div style="
                    height: 100%;
                    background: linear-gradient(90deg, #00d4ff, #00ff88);
                    border-radius: 2px;
                    animation: progress 2s ease-in-out infinite;
                "></div>
            </div>
        </div>
    </div>
    
    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        @keyframes progress {{
            0% {{ width: 0%; }}
            50% {{ width: 70%; }}
            100% {{ width: 100%; }}
        }}
    </style>
    """

def reset_watch_mode_keys():
    """Reset all watch mode keys to default (Youtube Online)"""
    # Don't reset if we're in a dialog context
    if 'dialog_context' in st.session_state:
        return
        
    # Get all session state keys that start with watch_mode
    keys_to_reset = [key for key in st.session_state.keys() if key.startswith("watch_mode_")]
    
    for key in keys_to_reset:
        del st.session_state[key]



@st.dialog("Đang tải video...")
def download_video_dialog(video_id: str, local_video_root: Path):
    """Show download dialog with loading animation and run download logic inside"""
    
    # Set dialog context to prevent watch mode reset
    st.session_state['dialog_context'] = True
    
    # Create placeholder for dynamic content
    content_placeholder = st.empty()
    
    # Show loading animation
    loading_message = f"File chưa có sẵn. Đang tải {video_id}.mp4 từ Kaggle..."
    content_placeholder.markdown(f"""
    <div style="
        text-align: center;
        padding: 40px 20px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    ">
        <div style="
            font-size: 60px;
            margin-bottom: 30px;
            animation: spin 1s linear infinite;
        ">⏳</div>
        <div style="font-size: 24px; font-weight: 600; color: #333; margin-bottom: 20px;">
            Đang tải video...
        </div>
        <div style="font-size: 18px; color: #666; margin-bottom: 30px;">
            {loading_message}
        </div>
        <div style="font-size: 16px; color: #999;">
            Vui lòng đợi trong giây lát...
        </div>
        <div style="
            margin-top: 40px;
            width: 100%;
            height: 8px;
            background: #f0f0f0;
            border-radius: 4px;
            overflow: hidden;
        ">
            <div style="
                height: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                border-radius: 4px;
                animation: progress 2s ease-in-out infinite;
            "></div>
        </div>
    </div>
    
    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        @keyframes progress {{
            0% {{ width: 0%; }}
            50% {{ width: 70%; }}
            100% {{ width: 100%; }}
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Run download logic inside dialog
    try:
        # Pre-flight checks
        try:
            import kagglehub
        except ImportError:
            content_placeholder.error("Lỗi: Thư viện 'kagglehub' chưa được cài đặt. Vui lòng chạy 'pip install kagglehub'.")
            return None

        kaggle_config_dir = Path.cwd() / ".kaggle"
        if not (kaggle_config_dir / "kaggle.json").exists():
            content_placeholder.error("Lỗi: Không tìm thấy file .kaggle/kaggle.json. Vui lòng đặt Kaggle API token của bạn vào đó.")
            return None
                        
        os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_config_dir)

        dataset_handle = "ndmhoang/hcm25-batch1-videos" if video_id.startswith("L") else "duyminhhong/hcmai25-batch2-videos"
        remote_file_path = f"video/{video_id}.mp4"
        local_video_root.mkdir(exist_ok=True)

        print(f"[KaggleHub] Downloading {remote_file_path} from {dataset_handle}...")
        
        # Download to system cache
        cached_path_str = kagglehub.dataset_download(dataset_handle, path=remote_file_path)
        cached_path = Path(cached_path_str)
        print(f"[KaggleHub] Downloaded to cache: {cached_path}")

        # Move the file from cache to our local ./videos directory
        local_video_path = local_video_root / f"{video_id}.mp4"
        shutil.move(cached_path, local_video_path)
        print(f"[KaggleHub] Moved to: {local_video_path}")

        # Show success message
        content_placeholder.markdown(f"""
        <div style="
            text-align: center;
            padding: 40px 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
            <div style="
                font-size: 60px;
                margin-bottom: 30px;
                animation: bounce 0.6s ease-in-out;
            ">✅</div>
            <div style="font-size: 24px; font-weight: 600; color: #333; margin-bottom: 20px;">
                Tải thành công!
            </div>
            <div style="font-size: 18px; color: #666;">
                Video đã được tải về thành công
            </div>
        </div>
        
        <style>
            @keyframes bounce {{
                0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
                40% {{ transform: translateY(-10px); }}
                60% {{ transform: translateY(-5px); }}
            }}
        </style>
        """, unsafe_allow_html=True)
        
        # Wait a bit then close dialog
        time.sleep(2.0)
        # Clear dialog context before rerun
        if 'dialog_context' in st.session_state:
            del st.session_state['dialog_context']
        st.rerun()  # Refresh the main UI to show the video player
        return local_video_path

    except Exception as e:
        error_message = str(e)
        print(f"[KaggleHub Error] {error_message}")
        
        # Show error message
        content_placeholder.markdown(f"""
        <div style="
            text-align: center;
            padding: 40px 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
            <div style="
                font-size: 60px;
                margin-bottom: 30px;
                animation: shake 0.5s ease-in-out;
            ">❌</div>
            <div style="font-size: 24px; font-weight: 600; color: #333; margin-bottom: 20px;">
                Tải thất bại!
            </div>
            <div style="font-size: 18px; color: #666;">
                Có lỗi xảy ra trong quá trình tải video
            </div>
        </div>
        
        <style>
            @keyframes shake {{
                0%, 100% {{ transform: translateX(0); }}
                25% {{ transform: translateX(-5px); }}
                75% {{ transform: translateX(5px); }}
            }}
        </style>
        """, unsafe_allow_html=True)
        
        # Wait a bit then close dialog
        time.sleep(3.0)
        # Clear dialog context before rerun
        if 'dialog_context' in st.session_state:
            del st.session_state['dialog_context']
        st.rerun()  # Refresh the main UI
        return None

def download_video_from_kaggle(video_id: str, local_video_root: Path, placeholder) -> Optional[Path]:
    """
    Downloads a video using the reliable kagglehub library, moves it from cache to the
    local videos folder, and shows a user-friendly dialog on the UI during the process.
    """
    local_video_path = local_video_root / f"{video_id}.mp4"
    if local_video_path.exists():
        placeholder.empty()
        return local_video_path
    
    # Show download dialog that runs download logic
    result = download_video_dialog(video_id, local_video_root)
    
    # Return the path if it exists after download
    if result and result.exists():
        return result
    else:
        return None


# ============================ LOCAL VIDEO SERVER ============================
def make_video_app(root_dir: str) -> Flask:
    """Creates a Flask app to serve video files with Range requests."""
    app = Flask(__name__)

    @app.after_request
    def add_cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Range, Origin, Content-Type, Accept"
        resp.headers["Access-Control-Expose-Headers"] = "Content-Range, Accept-Ranges, Content-Length"
        resp.headers["Accept-Ranges"] = "bytes"
        return resp

    @app.route("/video/<path:filename>")
    def serve_video(filename: str):
        filepath = os.path.join(root_dir, filename)
        if not os.path.isfile(filepath):
            return "Not found", 404
        return send_file(filepath, mimetype="video/mp4", as_attachment=False, conditional=True)
    return app

def ensure_server_once(app: Flask, host="127.0.0.1", port=8801):
    """Ensures the Flask server is started only once in a background thread."""
    if not st.session_state.get("_local_video_server_started"):
        threading.Thread(target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False), daemon=True).start()
        st.session_state["_local_video_server_started"] = True
        print(f"[Server] Local video server started on http://{host}:{port}")

# ============================ .env LOADING (no external deps) ============================
if os.path.exists('.env'):
    try:
        with open('.env', 'r', encoding='utf-8') as f:
            for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    k = k.strip(); v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

# ============================ DEFAULT CONFIG (as requested) ==============================
DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/shared/HCMC-AI/")
DEFAULT_TOP_K = 20
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

B1 = {
    "clip_dir": f"{DATA_ROOT}b1/clip-features-32",
    "map_csv": f"{DATA_ROOT}b1/map-keyframes",
    "kf_root": f"{DATA_ROOT}b1/keyframes",
    "vid_root": f"{DATA_ROOT}b1/video",
    "keep_dir": f"{DATA_ROOT}b1/clip_keep_idxs",
}

B2 = {
    "clip_dir": f"{DATA_ROOT}b2/clip-features-32",
    "map_csv": f"{DATA_ROOT}b2/map-keyframes",
    "kf_root": f"{DATA_ROOT}b2/keyframes",
    "vid_root": f"{DATA_ROOT}b2/video",
    "keep_dir": f"{DATA_ROOT}b2/clip_keep_idxs",
}

# Find BM25 database paths
def find_db_path(data_root, batch_name):
    """Find the BM25 database path for a given batch"""
    import glob

    # Try common locations
    possible_paths = [
        f"{data_root}{batch_name}/text_bm25.db",
        f"{data_root}{batch_name}/text_bm25.db",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path


B1_DB_PATH = find_db_path(DATA_ROOT, "b1")
B2_DB_PATH = find_db_path(DATA_ROOT, "b2")

DATASETS = {"b1", "b2"}
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MAX_TOKENS = 10000
# Use GOOGLE_API_KEY for Gemini
GEMINI_API_KEY = GOOGLE_API_KEY
USE_LLM_ASPECTS = bool(GEMINI_API_KEY)
WINDOW_SEC = 40.0
SKIP_SEC = 0.0
DELTA_SEC = 20.0
TOP_VIDEOS = 20
LIMIT_VIDEOS = None
EXCLUDE_SER = set()
LOCAL_VIDEO_ROOT = Path("./videos")
LOCAL_VIDEO_URL_BASE = "http://127.0.0.1:8801/video"

# ================================== UTILS ===============================================
def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32", copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def sec_to_mmss(t: float) -> str:
    if t < 0 or not np.isfinite(t):
        return "00:00"
    m = int(t // 60)
    s = int(round(t - m * 60))
    if s == 60:
        m, s = m + 1, 0
    return f"{m:02d}:{s:02d}"

def parse_mmss(s: str, default_sec: float = 0.0) -> float:
    """Parse 'mm:ss' or raw seconds string to float seconds. Returns default on error."""
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
    except Exception:
        return float(default_sec)

def dataset_for_video(video_id: str) -> str:
    return "b1" if video_id.startswith("L") else "b2"

def keyframe_path(video_id: str, frame_id: int) -> str:
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
    D = B1 if dataset_for_video(video_id) == "b1" else B2
    return f"{D['clip_dir']}/{video_id}.npy"

def map_csv_path_for_video(video_id: str) -> str:
    D = B1 if dataset_for_video(video_id) == "b1" else B2
    return f"{D['map_csv']}/{video_id}.csv"

def keep_idxs_path_for_video(video_id: str) -> str:
    D = B1 if dataset_for_video(video_id) == "b1" else B2
    return f"{D['keep_dir']}/{video_id}_keep_idxs.npy"

def safe_imread(path: str) -> Image.Image:
    try:
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
        else:
            # Return None if file doesn't exist, don't return fallback
            return None
    except Exception:
        return None

def timestamp_to_keyframe_number(vid: str, timestamp: float) -> int:
    """Convert timestamp to keyframe number using map-keyframes CSV"""
    try:
        # Try to read map-keyframes CSV for exact mapping
        D = B1 if dataset_for_video(vid) == "b1" else B2
        map_csv_path = f"{D['map_csv']}/{vid}.csv"

        if os.path.exists(map_csv_path):
            import pandas as pd
            df = pd.read_csv(map_csv_path)

            if 'pts_time' in df.columns and 'n' in df.columns:
                # Find the keyframe with timestamp closest to the target
                df['time_diff'] = abs(df['pts_time'] - timestamp)
                closest_row = df.loc[df['time_diff'].idxmin()]

                # Return the keyframe number (n column)
                keyframe_num = int(closest_row['n'])
                print(f"[Keyframe] Found exact mapping: timestamp {timestamp:.2f}s -> keyframe {keyframe_num}")
                return keyframe_num

    except Exception as e:
        print(f"[Keyframe] Error reading map CSV for {vid}: {e}")

    # Fallback: estimate based on typical keyframe extraction rate
    fps = get_fps_from_csv(vid) or 25.0
    estimated_frame = int(timestamp * fps)

    # For batch 2, keyframes seem to be more frequent (every ~1-2 seconds)
    if dataset_for_video(vid) == "b1":
        keyframe_interval = 25  # ~1 second at 25fps
    else:  # batch 2
        keyframe_interval = 50  # More conservative for batch 2

    estimated_keyframe = max(1, estimated_frame // keyframe_interval)
    print(f"[Keyframe] Using estimation: timestamp {timestamp:.2f}s -> estimated keyframe {estimated_keyframe}")

    return estimated_keyframe

def find_best_keyframe(vid: str, target_frame: int) -> tuple:
    """Find the best available keyframe closest to target_frame, using parquet data for filtering"""
    # Get keyframes data from parquet
    keyframes_data = get_keyframes_data()
    dataset = "b1" if vid.startswith("L") else "b2"

    # Try frames in order of preference
    frame_offsets = [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8, -9, 9, -10, 10]

    for offset in frame_offsets:
        frame = target_frame + offset
        if frame >= 0:
            try:
                # Check if this frame is in the canonical keyframes list
                if dataset in keyframes_data:
                    df = keyframes_data[dataset]
                    # Filter by video_id and frame number
                    matching_frames = df[(df['video_id'] == vid) & (df['n'] == frame)]
                    if not matching_frames.empty:
                        # Check if it's canonical (not duplicate)
                        if dataset == 'b2':
                            is_canonical = matching_frames['is_canonical'].iloc[0]
                            if not is_canonical:
                                continue  # Skip non-canonical frames

                img = safe_imread(keyframe_path(vid, frame))
                if img is not None:  # Valid image exists
                    return img, frame
            except Exception as e:
                print(f"[KEYFRAME] Error checking frame {frame} for {vid}: {e}")
                continue

    # If no frames found around target, try to find any available canonical frame
    # by scanning the keyframe directory
    D = B1 if dataset_for_video(vid) == "b1" else B2
    L, V = vid.split("_")
    kf_dir = f"{D['kf_root']}/{L}/{V}"

    if os.path.exists(kf_dir):
        try:
            from pathlib import Path
            # Get all image files in the directory
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(list(Path(kf_dir).glob(ext)))

            if image_files:
                # Sort by frame number (extract number from filename)
                def extract_frame_num(filepath):
                    try:
                        filename = filepath.stem
                        # Extract number from filename (remove leading zeros)
                        return int(filename.lstrip('0') or '0')
                    except (ValueError, AttributeError):
                        return 0

                image_files.sort(key=extract_frame_num)

                # Filter by canonical keyframes from parquet data
                canonical_frames = []
                try:
                    if dataset in keyframes_data:
                        df = keyframes_data[dataset]
                        video_frames = df[df['video_id'] == vid]

                        if dataset == 'b1':
                            # B1 uses all frames in keyframes_keepers
                            canonical_frames = video_frames['n'].tolist() if 'n' in video_frames.columns else []
                        else:
                            # B2 uses only canonical frames
                            if 'is_canonical' in video_frames.columns:
                                canonical_frames = video_frames[video_frames['is_canonical'] == True]['n'].tolist()
                            else:
                                canonical_frames = video_frames['n'].tolist() if 'n' in video_frames.columns else []
                except Exception as e:
                    print(f"[KEYFRAME] Error filtering canonical frames: {e}")
                    canonical_frames = []

                # Try the frame closest to target_frame from canonical list
                target_file = None
                min_diff = float('inf')

                for img_file in image_files:
                    frame_num = extract_frame_num(img_file)
                    # Only consider canonical frames
                    if canonical_frames and frame_num not in canonical_frames:
                        continue

                    diff = abs(frame_num - target_frame)
                    if diff < min_diff:
                        min_diff = diff
                        target_file = img_file

                if target_file:
                    img = safe_imread(str(target_file))
                    if img is not None:
                        frame_num = extract_frame_num(target_file)
                        return img, frame_num

                # If still no good image, try the last available canonical frame
                for img_file in reversed(image_files):
                    frame_num = extract_frame_num(img_file)
                    if not canonical_frames or frame_num in canonical_frames:
                        img = safe_imread(str(img_file))
                        if img is not None:
                            return img, frame_num

        except Exception as e:
            print(f"Error scanning keyframe directory {kf_dir}: {e}")

    # Ultimate fallback - try frame 1
    img = safe_imread(keyframe_path(vid, 1))
    if img and img.size[0] > 100:
        return img, 1

    # Final fallback - return placeholder
    return Image.fromarray(np.full((360, 640, 3), 200, dtype=np.uint8)), 1

def list_videos_from_dirs(clip_dirs: List[str],
                          limit: Optional[int] = None,
                          include_series: Optional[set] = None,
                          exclude_series: Optional[set] = None,
                          include_regex: Optional[str] = None,
                          exclude_regex: Optional[str] = None,
                          include_ids: Optional[set] = None,
                          exclude_ids: Optional[set] = None) -> List[str]:
    files = []
    for d in clip_dirs:
        if d and os.path.isdir(d):
            files.extend(glob.glob(os.path.join(d, "*.npy")))
    vids = [os.path.splitext(os.path.basename(p))[0] for p in sorted(set(files))]
    def keep(vid: str) -> bool:
        sid = vid.split("_", 1)[0]
        if include_series and sid not in include_series: return False
        if exclude_series and sid in exclude_series: return False
        if include_ids and vid not in include_ids: return False
        if exclude_ids and vid in exclude_ids: return False
        if include_regex and not re.search(include_regex, vid): return False
        if exclude_regex and re.search(exclude_regex, vid): return False
        ds = dataset_for_video(vid)
        if ds not in DATASETS: return False
        return True
    vids = [v for v in vids if keep(v)]
    return vids[:limit] if limit else vids

def load_times_for_video(video_id: str, n_feats: int) -> np.ndarray:
    csv_path = map_csv_path_for_video(video_id)
    if not os.path.exists(csv_path):
        return np.linspace(0.0, float(max(n_feats - 1, 1)), num=n_feats, dtype=float)
    df = pd.read_csv(csv_path)
    t_all = df["pts_time"].to_numpy().astype(float)
    keep_path = keep_idxs_path_for_video(video_id)
    if os.path.exists(keep_path):
        try:
            keep = np.load(keep_path).astype(int)
            t = t_all[np.clip(keep - 1, 0, len(t_all) - 1)]
            if len(t) == n_feats:
                return t
        except Exception:
            pass
    if len(t_all) == n_feats:
        return t_all
    t_end = float(t_all[-1]) if len(t_all) else float(n_feats - 1)
    return np.linspace(0.0, t_end, num=n_feats, dtype=float)

# =============================== CLIP TEXT ENCODER ======================================
_CLIP_RT = {"model": None, "tokenizer": None, "device": None, "backend": None}

def ensure_clip_text_encoder(model_name='ViT-B-32', pretrained='openai'):
    if _CLIP_RT["model"] is not None:
        return
    try:
        import open_clip, torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        model.to(device); model.eval()
        _CLIP_RT.update({"model": model, "tokenizer": tokenizer, "device": device, "backend": "open_clip"});
        return
    except Exception:
        pass
    try:
        import clip, torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load(model_name, device=device, jit=False)
        _CLIP_RT.update({"model": model, "tokenizer": clip.tokenize, "device": device, "backend": "clip"});
        return
    except Exception:
        raise RuntimeError("CLIP text encoder (open_clip/clip) not found.")

def encode_text(texts: List[str]) -> np.ndarray:
    ensure_clip_text_encoder()
    import torch
    toks = _CLIP_RT["tokenizer"](texts)
    with torch.no_grad():
        toks = toks.to(_CLIP_RT["device"])
        feats = _CLIP_RT["model"].encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().numpy().astype("float32")

# =============================== ASPECTS + SEEDS (Gemini) ================================
FALLBACK = {
    "aspects": [
        {"name": "grand sand sculpture", "prompts": [
            "grand sand sculpture at sand sculpture festival, monumental sand art, detailed figures",
            "large sand sculpture exhibit, festival display, intricate sand carving"
        ], "weight": 1.0},
        {"name": "rollerblader + two skateboarders", "prompts": [
            "youth street sports in sand sculpture: one rollerblader and two skateboarders",
            "skaters carved in sand: inline skates and two skateboards"
        ], "weight": 1.35},
        {"name": "arched backdrop with carved letters", "prompts": [
            "large arched backdrop with carved letters, inscriptions in sand",
            "curved arch structure with text engraved"
        ], "weight": 0.9},
        {"name": "multiple sand artworks", "prompts": [
            "multiple sand sculptures on display, exhibition with several sand artworks",
            "variety of sand art pieces shown together at festival"
        ], "weight": 0.8},
        {"name": "two pink smoke columns", "prompts": [
            "two pink smoke columns, pink smoke flares near sand sculptures",
            "pink colored smoke pillars beside sand sculpture"
        ], "weight": 1.35}
    ],
    "seeds": {
        "unigrams": ["điêu","khắc","cát","lễ","hội","tác","phẩm","trượt","patin","ván","skate","skateboard",
                      "vòm","cong","chữ","khói","hồng","cột","màu","thanh","niên"],
        "phrases":  ["điêu khắc cát","lễ hội điêu khắc","tác phẩm điêu khắc","trượt patin","trượt ván",
                      "vòm cong","khắc chữ","khói hồng","khói màu hồng","cột khói"]
    }
}

def extract_aspects_and_seeds_with_gemini(query_vi: str, temperature: float = 0.7, top_p: float = 0.9,
                                         model: str = "gemini-2.5-flash", max_aspects: str = "N") -> Dict:
    if not (USE_LLM_ASPECTS and GEMINI_API_KEY):
        print("[Gemini] Using FALLBACK aspects/seeds (no API key)")
        return {"aspects": FALLBACK.get("aspects", []), "seeds": FALLBACK.get("seeds", {}), "source": "fallback", "error": "no_api_key"}
    sys_inst = """
Bạn là công cụ xây dựng truy vấn cho VIDEO RETRIEVAL / TRANSCRIPT SEARCH.

NHIỆM VỤ
- Đầu vào (INPUT): một mô tả TỰ NHIÊN (tiếng Việt hoặc tiếng Anh) về nội dung video cần tìm.
- Đầu ra (OUTPUT): MỘT JSON HỢP LỆ duy nhất, gồm 2 trường: "aspects" và "seeds".
- Tuyệt đối KHÔNG thêm giải thích, không thêm bình luận, không Markdown.

RÀNG BUỘC ĐẦU RA
- JSON dùng dấu ngoặc kép tiêu chuẩn (").
- "aspects": mảng TỐI ĐA {max_aspects} phần tử. Mỗi phần tử là:
  {{
    "name": "<tên khía cạnh bằng TIẾNG VIỆT, ngắn gọn, phân biệt mạnh>",
    "prompts": ["<2-4 prompt bằng TIẾNG ANH, mô tả rõ hình/khung cảnh/hành động>", ...],
    "weight": <số thực từ 0.80 đến 1.30, 2 chữ số thập phân, phản ánh MỨC ĐỘ PHÂN BIỆT>
  }}
- "seeds": {{
    "unigrams": [<= 28 từ khóa đơn TIẾNG VIỆT, không stopword phổ biến, ưu tiên đặc trưng],
    "phrases":  [<= 18 cụm từ TIẾNG VIỆT (2–5 từ), mô tả giàu ngữ nghĩa/đặc trưng]
  }}
- Không vượt quá giới hạn số lượng. Không khóa học thuật/lý thuyết. Không trộn ngôn ngữ sai yêu cầu:
  • aspects.name = VI | aspects.prompts = EN
  • seeds.* = VI
- Loại trùng lặp, chuẩn hóa khoảng trắng, giữ số/đơn vị nếu phân biệt mạnh (vd: "211kg").
- Ưu tiên thuật ngữ/đặc trưng: thực thể, đồ vật, màu sắc, hành động, bối cảnh, đạo cụ, biểu tượng, con số hiếm.
- Weight: 1.00 là trung tính. >1.15 cho khía cạnh "rất chẩn đoán" (độc nhất hiếm gặp). 0.90–0.99 cho khía cạnh phụ trợ/ít chắc chắn. Không dưới 0.80, không trên 1.30.

CHIẾN LƯỢC TÓM TẮT
1) Tách khung cảnh thành 4–7 khía cạnh độc lập, mỗi khía cạnh có 2–4 prompt EN miêu tả rõ ràng (phù hợp cả BM25/CLIP).
2) Seeds VI:
   - unigrams: danh từ/chủ thể/màu/sự kiện/đạo cụ (tránh hư từ). Ưu tiên từ phân biệt cao.
   - phrases: kết hợp 2–5 từ tạo cụm đặc trưng (vd: "người phụ nữ cầm sen", "treo trên xe cẩu").
3) Không suy diễn ngoài mô tả. Nếu mơ hồ, chọn phương án an toàn, vẫn điền JSON đủ cấu trúc.

ĐỊNH DẠNG FEW-SHOT (tham khảo)
- Mỗi ví dụ gồm khối "INPUT:" và "OUTPUT:". Học cách cấu trúc và độ chi tiết từ các ví dụ.
- Khi xử lý mô tả mới, CHỈ trả JSON như phần "OUTPUT" (không in "INPUT/OUTPUT").

==================== FEW-SHOT EXAMPLES ====================

INPUT:
Trong một lễ hội trình diễn, các người mẫu lần lượt bước trên sàn. Mở màn là những bộ cổ phục "Nhật Bình" với đủ màu sắc. Trong đó, có cảnh người phụ nữ cầm những bông sen đứng đối diện nam người mẫu và cả 2 mặc áo màu xanh lá. Sau đó là các trang phục hiện đại hơn. Kết thúc bằng cảnh các người mẫu đồng loạt ra đứng sân khấu chào khán giả.

OUTPUT:
{{
  "aspects": [
    {{
      "name": "Cổ phục Nhật Bình",
      "prompts": [
        "Imperial Vietnamese 'Nhat Binh' robe on runway",
        "traditional court costume showcased on catwalk",
        "models wearing royal Vietnamese attire in a fashion show"
      ],
      "weight": 1.28
    }},
    {{
      "name": "Người phụ nữ cầm bông sen",
      "prompts": [
        "woman holding lotus flowers on stage",
        "female model holding lotus facing a male model"
      ],
      "weight": 1.25
    }},
    {{
      "name": "Tông màu xanh lá đồng bộ",
      "prompts": [
        "matching green outfits on both models",
        "green clothing theme during runway scene"
      ],
      "weight": 1.18
    }},
    {{
      "name": "Sàn catwalk / trình diễn thời trang",
      "prompts": [
        "fashion runway catwalk sequence",
        "models walking the stage in a fashion festival",
        "opening sequence of a fashion show"
      ],
      "weight": 1.05
    }},
    {{
      "name": "Chào kết đồng loạt",
      "prompts": [
        "final bow with all models on stage",
        "group lineup at the end of the show"
      ],
      "weight": 1.10
    }}
  ],
  "seeds": {{
    "unigrams": [
      "lễ hội","trình diễn","sàn diễn","catwalk","người mẫu",
      "cổ phục","nhật bình","áo xanh","xanh lá","bông sen",
      "cầm sen","đối diện","nam người mẫu","mở màn","hiện đại",
      "kết thúc","đồng loạt","chào khán giả","sân khấu"
    ],
    "phrases": [
      "áo nhật bình",
      "người phụ nữ cầm sen",
      "hai người mẫu áo xanh",
      "đối diện nam người mẫu",
      "mở màn cổ phục",
      "trang phục hiện đại hơn",
      "dàn người mẫu chào khán giả",
      "bước trên sàn diễn"
    ]
  }}
}}

----------------------------------------------------------

INPUT:
Đoạn video đưa tin về việc ngư dân vừa câu được một con cá nhám với kích thước hiếm gặp, nặng đến 211kg. Đoạn tin đưa hình ảnh của con cá rất lớn treo trên một chiếc xe cẩu.

OUTPUT:
{{
  "aspects": [
    {{
      "name": "Cá nhám khổng lồ 211kg",
      "prompts": [
        "massive shark weighing 211kg",
        "giant shark caught by fishermen in a news report"
      ],
      "weight": 1.30
    }},
    {{
      "name": "Treo trên xe cẩu",
      "prompts": [
        "large fish hanging on a crane truck",
        "crane lifting the catch in reportage footage"
      ],
      "weight": 1.22
    }},
    {{
      "name": "Ngư dân / bản tin thời sự",
      "prompts": [
        "news segment about fishermen at a port",
        "reportage coverage of a fishing catch"
      ],
      "weight": 1.00
    }},
    {{
      "name": "Kích thước hiếm gặp",
      "prompts": [
        "rare unusually large specimen",
        "record-sized fish highlighted in news"
      ],
      "weight": 1.12
    }}
  ],
  "seeds": {{
    "unigrams": [
      "ngư dân","cá nhám","cá mập","khổng lồ","211kg",
      "hiếm gặp","treo","xe cẩu","đưa tin","bản tin",
      "con cá","kích thước","câu được","cảng","trọng lượng",
      "hiếm","xe tải cẩu"
    ],
    "phrases": [
      "cá nhám nặng 211kg",
      "cá mập kích thước hiếm gặp",
      "treo trên xe cẩu",
      "đưa tin ngư dân",
      "hình ảnh con cá rất lớn"
    ]
  }}
}}

----------------------------------------------------------

INPUT:
Đoạn video đưa tin về một cuộc đua giữa các kị sĩ. Một trong các phân cảnh là hình ảnh đồng cỏ xanh rộng, các kị sĩ cưỡi ngựa chạy trên cánh đồng này. Rất đông người đứng xem xung quanh khu vực đua. Trên đường đua có cắm những lá cờ màu đỏ.

OUTPUT:
{{
  "aspects": [
    {{
      "name": "Cuộc đua kị sĩ / đua ngựa",
      "prompts": [
        "horse race with jockeys",
        "riders on horseback during competition"
      ],
      "weight": 1.28
    }},
    {{
      "name": "Đồng cỏ xanh rộng",
      "prompts": [
        "wide green meadow landscape",
        "horses running across open field"
      ],
      "weight": 1.15
    }},
    {{
      "name": "Khán giả đông xung quanh",
      "prompts": [
        "crowd of spectators watching the race",
        "many people standing around the track"
      ],
      "weight": 1.10
    }},
    {{
      "name": "Cờ đỏ dọc đường đua",
      "prompts": [
        "red flags along the race track",
        "track markers with red flags"
      ],
      "weight": 1.20
    }}
  ],
  "seeds": {{
    "unigrams": [
      "kị sĩ","đua ngựa","đồng cỏ","xanh","chạy",
      "đường đua","cờ đỏ","khán giả","đông người","cánh đồng",
      "cưỡi ngựa","cuộc đua","khu vực đua","lá cờ","xung quanh"
    ],
    "phrases": [
      "các kị sĩ cưỡi ngựa",
      "chạy trên đồng cỏ xanh",
      "cờ đỏ dọc đường đua",
      "rất đông người đứng xem",
      "cuộc đua kị sĩ"
    ]
  }}
}}

================== HẾT FEW-SHOT EXAMPLES ==================

QUY TRÌNH KHI NHẬN ĐẦU VÀO MỚI
1) Phân rã mô tả thành khía cạnh (chủ thể, hành động, bối cảnh, màu sắc, đạo cụ, con số).
2) Tạo 2–4 prompts EN/khía cạnh, mô tả rõ ràng, có ích cho cả CLIP/Caption retrieval.
3) Sinh seeds VI:
   - unigrams: danh từ/thuật ngữ phân biệt, giữ số/đơn vị (vd: "211kg"), tránh hư từ.
   - phrases: 2–5 từ, mô tả tổ hợp giàu thông tin (chủ thể + hành động + bối cảnh/màu/đạo cụ).
4) Đặt weight 0.80–1.30 theo độ chẩn đoán/độ chắc chắn.
5) Xuất MỘT JSON HỢP LỆ, không thêm bất cứ ký tự/giải thích nào khác.
""".format(max_aspects=max_aspects)

    payload = {"query_vi": query_vi}
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        try:
            resp = client.responses.generate(
            model=GEMINI_MODEL,
                input=json.dumps(payload, ensure_ascii=False),
                system_instruction=sys_inst,
                config={"max_output_tokens": GEMINI_MAX_TOKENS,
                        "response_mime_type": "application/json"}
            )
            txt = getattr(resp, "output_text", None)
            if not txt:
                raise RuntimeError("Empty output_text")
        except Exception:
            cfg = genai.types.GenerateContentConfig(
                max_output_tokens=GEMINI_MAX_TOKENS,
                response_mime_type="application/json",
                system_instruction=sys_inst,
                temperature=temperature,
                top_p=top_p
            )
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=json.dumps(payload, ensure_ascii=False),
                config=cfg
        )
            txt = getattr(resp, "text", None) or getattr(resp, "output_text", None)
        if not txt:
                raise RuntimeError("Empty text")
        data = json.loads(txt)
        if "aspects" not in data or not isinstance(data["aspects"], list):
            raise ValueError("no aspects")
        if "seeds" not in data or not isinstance(data["seeds"], dict):
            data["seeds"] = {"unigrams": [], "phrases": []}
        data["source"] = "gemini"
        return data
    except Exception as e:
        print(f"[Gemini] Call failed: {e}")
        print("[Gemini] Using fallback translation method...")
        aspects, seeds = fallback_translate_and_create_aspects(query_vi)
        return {"aspects": aspects, "seeds": seeds, "source": "fallback", "error": str(e)}

# =============================== BUILD QUERY VECTORS ====================================

def build_query_vecs_for_aspects(aspects: List[Dict]) -> Tuple[List[np.ndarray], List[str], List[float]]:
    all_prompts, ranges, names, weights = [], [], [], []
    for a in aspects:
        ps = list(dict.fromkeys(a.get("prompts", [])))
        s = len(all_prompts)
        all_prompts.extend(ps)
        ranges.append((s, len(all_prompts)))
        names.append(a.get("name", "aspect"))
        weights.append(float(a.get("weight", 1.0)))
    if not all_prompts:
        all_prompts = ["generic scene"]; ranges = [(0, 1)]; names = ["generic"]; weights = [1.0]
    q = encode_text(all_prompts).astype("float32")
    q = l2_normalize_rows(q)
    q_per_aspect = [q[s:e] for (s, e) in ranges]
    return q_per_aspect, names, weights

# =============================== WINDOW SCORING (CLIP) ==================================

def best_window_multi_aspect(feats: np.ndarray, times: np.ndarray,
                             q_per_aspect: List[np.ndarray],
                             aspect_weights: List[float],
                             window_sec: float = 40.0,
                             min_start_sec: float = 0.0):
    N = feats.shape[0]
    if N == 0:
        return {"score": -1e9, "i0": 0, "i1": 0, "t0": 0.0, "t1": 0.0,
                "aspect_frames": [1] * len(q_per_aspect), "aspect_scores": [0.0] * len(q_per_aspect)}
    start_idx = int(np.searchsorted(times, min_start_sec, side="left"))
    start_idx = min(max(start_idx, 0), N - 1)
    feats = feats[start_idx:]; times = times[start_idx:]; Nsub = feats.shape[0]
    sims_list = []
    for qA in q_per_aspect:
        sims = (qA @ feats.T) if len(qA) > 0 else np.zeros((1, Nsub), dtype="float32")
        sims_list.append(sims.max(axis=0).astype("float32"))
    deques = [deque() for _ in range(len(sims_list))]
    best = {"score": -1e9, "i0": 0, "i1": 0, "peaks": [(0, 0.0)] * len(sims_list)}
    e = 0
    for i in range(Nsub):
        while e < Nsub and (times[e] - times[i] <= window_sec):
            for j, s in enumerate(sims_list):
                dq = deques[j]; val = float(s[e])
                while dq and dq[-1][1] <= val:
                    dq.pop()
                dq.append((e, val))
            e += 1
        win_sum, peaks = 0.0, []
        for j, dq in enumerate(deques):
            idx, val = (dq[0] if dq else (i, float("-inf")))
            peaks.append((idx, val)); win_sum += float(aspect_weights[j]) * val
        if win_sum > best["score"]:
            best = {"score": win_sum, "i0": i, "i1": e - 1, "peaks": peaks}
        for dq in deques:
            if dq and dq[0][0] == i:
                dq.popleft()
    aspect_frames = [int(start_idx + idx + 1) for (idx, _v) in best["peaks"]]
    aspect_scores = [float(_v) for (_idx, _v) in best["peaks"]]
    i0, i1 = best["i0"], best["i1"]
    t0 = float(times[i0]); t1 = float(times[min(i1, len(times) - 1)])
    return {"score": float(best["score"]),
            "i0": int(start_idx + i0), "i1": int(start_idx + i1),
            "t0": t0, "t1": t1,
            "aspect_frames": aspect_frames, "aspect_scores": aspect_scores}

def score_video_clip_window(video_id: str,
                            q_per_aspect: List[np.ndarray],
                            aspect_weights: List[float],
                            window_sec=40.0, min_start_sec=0.0):
    npy_path = npy_path_for_video(video_id)
    if not os.path.exists(npy_path):
        return {"video_id": video_id, "ok": False, "error": "no npy"}
    feats = np.load(npy_path).astype("float32")
    feats = l2_normalize_rows(feats)
    times = load_times_for_video(video_id, len(feats))
    win = best_window_multi_aspect(feats, times, q_per_aspect, aspect_weights,
                                   window_sec=window_sec, min_start_sec=min_start_sec)
    return {"video_id": video_id, "ok": True, "total": win["score"], "win": win}

def rank_videos_clip_window(aspects: List[Dict],
                            limit_videos=None, window_sec=40.0, min_start_sec=0.0,
                            include_series: set | None = None,
                            exclude_series: set | None = None,
                            include_regex: str | None = None,
                            exclude_regex: str | None = None,
                            include_ids: set | None = None,
                            exclude_ids: set | None = None):
    clip_dirs = []
    if "b1" in DATASETS: clip_dirs.append(B1["clip_dir"])
    if "b2" in DATASETS: clip_dirs.append(B2["clip_dir"])
    vids = list_videos_from_dirs(
        clip_dirs,
        limit=limit_videos,
        include_series=include_series, exclude_series=exclude_series,
        include_regex=include_regex, exclude_regex=exclude_regex,
        include_ids=include_ids, exclude_ids=exclude_ids
    )
    print(f"[CLIP-RANK] Found {len(vids)} videos to score from datasets: {DATASETS}")
    q_per_aspect, names, weights = build_query_vecs_for_aspects(aspects)
    out = []
    scored_count = 0
    for vid in vids:
        r = score_video_clip_window(vid, q_per_aspect, weights, window_sec=window_sec, min_start_sec=min_start_sec)
        scored_count += 1
        if r["ok"]:
            out.append(r)
    print(f"[CLIP-RANK] Successfully scored {scored_count} videos, {len(out)} returned valid results")
    out.sort(key=lambda x: x["total"], reverse=True)
    return out, names, weights

# ============================== TEXT RERANK (ULTRA-WIN) =================================

def _tok_vi(s):
    return re.findall(r"[0-9]+|[A-Za-zÀ-ỹĐđ]+", s.lower())

def build_keyword_sets(query_vi: str, seeds: Dict, max_unigrams=28, max_phrases=18):
    toks = _tok_vi(query_vi)
    unigrams = {t for t in toks if len(t) >= 2}
    phrases = {" ".join(toks[i:i + 2]) for i in range(len(toks) - 1)}
    unigrams.update(seeds.get("unigrams", []))
    phrases.update(seeds.get("phrases", []))
    uni = sorted(list(unigrams))[:max_unigrams]
    phr = sorted(list(phrases))[:max_phrases]
    q_any = " OR ".join(uni + [f'"{p}"' for p in phr]) if (uni or phr) else None
    q_phr = " OR ".join([f'"{p}"' for p in phr]) if phr else None
    return uni, phr, q_any, q_phr

def _connect_ro(db_path: str):
    return sqlite3.connect(db_path, check_same_thread=False)

def _count_one(db_path: str, vid: str, t0: float, t1: float, q_any: str | None, q_phr: str | None):
    conn = _connect_ro(db_path)
    cur = conn.cursor()
    sql = "SELECT COUNT(*) FROM docs WHERE video_id=? AND t BETWEEN ? AND ? AND docs MATCH ?"
    h_any = h_phr = 0
    if q_any:
        h_any = cur.execute(sql, (vid, t0, t1, q_any)).fetchone()[0]
    if q_phr:
        h_phr = cur.execute(sql, (vid, t0, t1, q_phr)).fetchone()[0]
    conn.close()
    return (vid, int(h_any), int(h_phr))

def _alias_windows_and_dbpaths(windows):
    g = {"b1": [], "b2": []}
    for w in windows:
        (g["b1"] if w["video_id"].startswith("L") else g["b2"]).append(w)
    out = []
    if g["b1"] and os.path.exists(B1_DB_PATH): out.append(("b1", B1_DB_PATH, g["b1"]))
    if g["b2"] and os.path.exists(B2_DB_PATH): out.append(("b2", B2_DB_PATH, g["b2"]))
    return out

def highlight_keywords_in_text(text: str, query: str) -> str:
    """Highlight keywords from BM25 query in the text"""
    import re

    # Type checking and conversion
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    if not isinstance(query, str):
        query = str(query) if query is not None else ""

    # If either is empty, return original text
    if not text or not query:
        return text

    try:
        # Get keywords from the original query (before conversion)

        # First, extract quoted phrases (e.g., "công an")
        quoted_phrases = re.findall(r'"([^"]+)"', query)
        quoted_phrases = [phrase.strip() for phrase in quoted_phrases if phrase.strip()]

        # Remove quoted phrases from query for further processing
        query_without_quotes = re.sub(r'"[^"]*"', '', query)

        # Split by operators and clean up - keep single characters and numbers
        keywords = re.split(r'\s*(?:AND|OR|NOT|\(|\))\s*', query_without_quotes)
        keywords = [k.strip() for k in keywords if k.strip()]

        # Also split by spaces and punctuation for individual words
        words = re.findall(r'\b\w+\b', query_without_quotes.lower())

        # Add numbers specifically (including single digits)
        numbers = re.findall(r'\b\d+\b', query_without_quotes)

        # Combine all keywords
        all_keywords = quoted_phrases + keywords + words + numbers

        # Remove duplicates while preserving order
        all_keywords = list(dict.fromkeys(all_keywords))

        # Allow all non-empty keywords
        all_keywords = [k for k in all_keywords if k.strip()]

        # Remove duplicates and sort by length (longest first) to avoid partial matches
        all_keywords = list(dict.fromkeys(all_keywords))  # Remove duplicates while preserving order
        all_keywords.sort(key=len, reverse=True)

        # Create highlighted text
        highlighted_text = text

        for keyword in all_keywords:
            # Create pattern based on keyword type
            if keyword.isdigit():
                # For numbers, use word boundaries but allow more flexible matching
                pattern = re.compile(r'(?<![a-zA-Z])' + re.escape(keyword) + r'(?![a-zA-Z])(?![^<]*</b>)', re.IGNORECASE)
            elif ' ' in keyword:
                # For phrases, match the exact sequence with word boundaries
                # Split the phrase and create a pattern that allows flexible whitespace
                words = keyword.split()
                if len(words) > 1:
                    # Create pattern with flexible whitespace between words
                    pattern_parts = [re.escape(word) for word in words]
                    flexible_pattern = r'\s+'.join(pattern_parts)
                    pattern = re.compile(r'\b' + flexible_pattern + r'\b(?![^<]*</b>)', re.IGNORECASE)
                else:
                    # Single word with space (shouldn't happen, but handle it)
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b(?![^<]*</b>)', re.IGNORECASE)
            else:
                # For single words, use word boundaries
                pattern = re.compile(r'\b' + re.escape(keyword) + r'\b(?![^<]*</b>)', re.IGNORECASE)

            # Replace all occurrences, preserving original case
            def replace_func(match):
                matched_text = match.group(0)
                return f'<b>{matched_text}</b>'

            highlighted_text = pattern.sub(replace_func, highlighted_text)

        return highlighted_text

    except Exception as e:
        print(f"[HIGHLIGHT] Error highlighting text: {e}")
        return text  # Return original text if highlighting fails

def convert_bm25_syntax(query: str) -> str:
    """Convert user-friendly BM25 syntax to SQLite FTS5 syntax"""
    import re

    # Replace operators
    # & -> AND
    query = re.sub(r'\s*&\s*', ' AND ', query)
    # | -> OR
    query = re.sub(r'\s*\|\s*', ' OR ', query)
    # - -> NOT (but need to be careful with positioning)
    query = re.sub(r'\s*-\s*', ' NOT ', query)

    # Handle phrases in quotes - keep as is
    # SQLite FTS5 supports "phrase search" directly

    # Clean up extra spaces
    query = re.sub(r'\s+', ' ', query).strip()

    return query

def load_keyframes_data():
    """Load keyframes data from parquet files for duplicate filtering"""
    keyframes_data = {}

    try:
        import pandas as pd

        # Load b1 keyframes keepers
        b1_path = f"{DATA_ROOT}b1/keyframes_keepers.parquet"
        if os.path.exists(b1_path):
            try:
                df_b1 = pd.read_parquet(b1_path)
                # Ensure column names are strings
                df_b1.columns = df_b1.columns.astype(str)
                keyframes_data['b1'] = df_b1
                print(f"[PARQUET] Loaded {len(df_b1)} keyframes for B1")
            except Exception as e:
                print(f"[PARQUET] Error loading B1 keyframes: {e}")

        # Load b2 manifest dedup
        b2_path = f"{DATA_ROOT}b2/manifest_dedup.parquet"
        if os.path.exists(b2_path):
            try:
                df_b2 = pd.read_parquet(b2_path)
                # Ensure column names are strings
                df_b2.columns = df_b2.columns.astype(str)
                # Filter only canonical keyframes (not duplicates)
                if 'is_canonical' in df_b2.columns:
                    df_b2_canonical = df_b2[df_b2['is_canonical'] == True]
                else:
                    df_b2_canonical = df_b2
                keyframes_data['b2'] = df_b2_canonical
                print(f"[PARQUET] Loaded {len(df_b2_canonical)} canonical keyframes for B2")
            except Exception as e:
                print(f"[PARQUET] Error loading B2 manifest: {e}")

        # Also load duplicates for reference
        dup_path = f"{DATA_ROOT}b1/duplicates.parquet"
        if os.path.exists(dup_path):
            try:
                df_dup = pd.read_parquet(dup_path)
                df_dup.columns = df_dup.columns.astype(str)
                print(f"[PARQUET] Loaded {len(df_dup)} duplicate records")
            except Exception as e:
                print(f"[PARQUET] Error loading duplicates: {e}")

        phash_path = f"{DATA_ROOT}b1/phash.parquet"
        if os.path.exists(phash_path):
            try:
                df_phash = pd.read_parquet(phash_path)
                df_phash.columns = df_phash.columns.astype(str)
                print(f"[PARQUET] Loaded {len(df_phash)} perceptual hash records")
            except Exception as e:
                print(f"[PARQUET] Error loading phash: {e}")

    except Exception as e:
        print(f"[PARQUET] Error loading parquet data: {e}")

    return keyframes_data

# Global variable to cache keyframes data
_KEYFRAMES_CACHE = None

def get_keyframes_data():
    """Get cached keyframes data"""
    global _KEYFRAMES_CACHE
    if _KEYFRAMES_CACHE is None:
        _KEYFRAMES_CACHE = load_keyframes_data()
    return _KEYFRAMES_CACHE

def apply_search_filters(results, exclude_series_list=None, exclude_videos_list=None):
    """Apply filters to search results (both BM25 and CLIP)"""
    if not exclude_series_list and not exclude_videos_list:
        return results

    filtered_results = []

    for result in results:
        video_id = result["video_id"]

        # Check series exclusion
        if exclude_series_list:
            series_id = video_id.split('_')[0]  # Extract series (e.g., L01 from L01V001)
            if series_id in exclude_series_list:
                print(f"[FILTER] Excluded {video_id} (series {series_id})")
                continue

        # Check video exclusion
        if exclude_videos_list and video_id in exclude_videos_list:
            print(f"[FILTER] Excluded {video_id} (specific video)")
            continue

        # Keep the result if it passes all filters
        filtered_results.append(result)

    print(f"[FILTER] Filtered {len(results)} -> {len(filtered_results)} results")
    return filtered_results

def deduplicate_bm25_results(results, time_threshold=30.0):
    """Remove duplicate results that are too close in time from the same video"""
    if not results:
        return results

    # Group by video_id
    video_groups = {}
    for result in results:
        video_id = result["video_id"]
        if video_id not in video_groups:
            video_groups[video_id] = []
        video_groups[video_id].append(result)

    deduplicated = []

    for video_id, video_results in video_groups.items():
        if len(video_results) <= 1:
            # Only one result for this video, keep it
            deduplicated.extend(video_results)
            continue

        # Sort by score (best first) and timestamp
        video_results.sort(key=lambda x: (x["score"], x["timestamp"]))

        # Keep the best result and filter others by time threshold
        selected_results = [video_results[0]]  # Always keep the best result

        for result in video_results[1:]:
            # Check if this result is far enough from all selected results
            too_close = False
            for selected in selected_results:
                time_diff = abs(result["timestamp"] - selected["timestamp"])
                if time_diff < time_threshold:  # Within threshold (default 30 seconds)
                    too_close = True
                    break

            if not too_close:
                selected_results.append(result)

        deduplicated.extend(selected_results)
        print(f"[DEDUPE] Video {video_id}: {len(video_results)} -> {len(selected_results)} results")

    # Sort final results by score
    deduplicated.sort(key=lambda x: x["score"])

    print(f"[DEDUPE] Total deduplicated: {len(results)} -> {len(deduplicated)} results")
    return deduplicated

def bm25_search_videos(query: str, limit: int = 50):
    """Search videos using BM25 full-text search"""
    import sqlite3

    # Validate query
    if not query or not query.strip():
        print("[BM25] Empty query provided")
        return []

    # Convert user-friendly syntax to SQLite FTS5 syntax
    converted_query = convert_bm25_syntax(query.strip())
    print(f"[BM25] Original query: '{query}'")
    print(f"[BM25] Converted query: '{converted_query}'")

    # Check if converted query is valid
    if not converted_query or not converted_query.strip():
        print("[BM25] Converted query is empty")
        return []

    results = []

    # Store original query for highlighting
    original_query = query

    # Search both databases
    for db_name, db_path in [("b1", B1_DB_PATH), ("b2", B2_DB_PATH)]:
        if not os.path.exists(db_path):
            print(f"[BM25] Database not found: {db_path}")
            continue

        conn = None
        cur = None
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()

            # BM25 search query - get full text from text column
            # Note: In FTS5, both snippet() and bm25() use table name, not column name
            sql = """
            SELECT video_id,
                   snippet(docs, 2, '<b>', '</b>', '...', 50) as snippet,
                   text as full_text,
                   bm25(docs) as score,
                   t as timestamp
            FROM docs
            WHERE text MATCH ?
            ORDER BY bm25(docs) DESC
            LIMIT ?
            """

            print(f"[BM25] Executing query for {db_name}: {converted_query}")
            cur.execute(sql, (converted_query, limit))
            rows = cur.fetchall()

            print(f"[BM25] Found {len(rows)} results in {db_name}")

            for row in rows:
                video_id, snippet, full_text, score, timestamp = row

                # Ensure all values are strings (SQLite might return integers for some fields)
                video_id = str(video_id) if video_id is not None else ""
                snippet = str(snippet) if snippet is not None else ""
                full_text = str(full_text) if full_text is not None else ""
                timestamp = float(timestamp) if timestamp is not None else 0.0

                # Highlight keywords in full text
                try:
                    highlighted_full_text = highlight_keywords_in_text(full_text, original_query)
                except Exception as e:
                    print(f"[BM25] Error highlighting text for {video_id}: {e}")
                    highlighted_full_text = full_text  # Use original text if highlighting fails

                results.append({
                    "video_id": video_id,
                    "snippet": snippet,
                    "full_text": full_text,
                    "highlighted_full_text": highlighted_full_text,
                    "score": float(score) if score is not None else 0.0,
                    "timestamp": timestamp,
                    "db_source": db_name
                })

        except Exception as e:
            print(f"[BM25] Error searching {db_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Ensure connections are properly closed
            try:
                if cur:
                    cur.close()
            except Exception as e:
                print(f"[BM25] Error closing cursor for {db_name}: {e}")
            try:
                if conn:
                    conn.close()
            except Exception as e:
                print(f"[BM25] Error closing connection for {db_name}: {e}")

    # Sort by score and return top results
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]

def transcript_seed_search(
    query_vi: str,
    seeds: dict,
    top_candidates: int = 100,
    delta_sec: float = 10.0,
    text_weights: dict | None = None,
    verbose: bool = True, timing: bool = True,
    *,
    workers: int = 8,
    phrases_only: bool = False,
    max_unigrams: int = 16, max_phrases: int = 10
):
    """
    Search videos using only transcript seeds via BM25 database (no CLIP scores).
    This is a pure text-based search using Gemini-generated seeds.
    """
    if text_weights is None:
        text_weights = {"any": 0.3, "phr": 0.7}
    t0 = time.perf_counter()
    
    print(f"[TRANSCRIPT-SEED] Searching BM25 databases with seeds: {seeds}")
    
    # Build search queries from seeds
    q_any = seeds.get("unigrams", [])
    q_phr = seeds.get("phrases", [])
    
    if not q_any and not q_phr:
        print("[TRANSCRIPT-SEED] No seeds provided, returning empty results")
        return pd.DataFrame(columns=["video_id", "hits_any", "hits_phr", "text_score", "fused", "t0", "t1"])
    
    # Search BM25 databases
    all_results = []
    
    # Search in Batch 1 database
    if "b1" in DATASETS:
        db_path = find_db_path(DATA_ROOT, "b1")
        if db_path and os.path.exists(db_path):
            print(f"[TRANSCRIPT-SEED] Searching Batch 1 database: {db_path}")
            try:
                conn = sqlite3.connect(db_path, check_same_thread=False)
                cursor = conn.cursor()
                
                # Search for unigrams
                for unigram in q_any:
                    try:
                        cursor.execute("SELECT video_id, text FROM transcripts WHERE text MATCH ?", (unigram,))
                        rows = cursor.fetchall()
                        for video_id, text in rows:
                            hits = text.lower().count(unigram.lower())
                            all_results.append({
                                "video_id": video_id,
                                "seed": unigram,
                                "seed_type": "unigram",
                                "hits": hits
                            })
                    except Exception as e:
                        print(f"[TRANSCRIPT-SEED] Error searching unigram '{unigram}': {e}")
                
                # Search for phrases
                for phrase in q_phr:
                    try:
                        cursor.execute("SELECT video_id, text FROM transcripts WHERE text MATCH ?", (f'"{phrase}"',))
                        rows = cursor.fetchall()
                        for video_id, text in rows:
                            hits = text.lower().count(phrase.lower())
                            all_results.append({
                                "video_id": video_id,
                                "seed": phrase,
                                "seed_type": "phrase",
                                "hits": hits
                            })
                    except Exception as e:
                        print(f"[TRANSCRIPT-SEED] Error searching phrase '{phrase}': {e}")
                
                cursor.close()
                conn.close()
            except Exception as e:
                print(f"[TRANSCRIPT-SEED] Error accessing Batch 1 database: {e}")
    
    # Search in Batch 2 database
    if "b2" in DATASETS:
        db_path = find_db_path(DATA_ROOT, "b2")
        if db_path and os.path.exists(db_path):
            print(f"[TRANSCRIPT-SEED] Searching Batch 2 database: {db_path}")
            try:
                conn = sqlite3.connect(db_path, check_same_thread=False)
                cursor = conn.cursor()
                
                # Search for unigrams
                for unigram in q_any:
                    try:
                        cursor.execute("SELECT video_id, text FROM transcripts WHERE text MATCH ?", (unigram,))
                        rows = cursor.fetchall()
                        for video_id, text in rows:
                            hits = text.lower().count(unigram.lower())
                            all_results.append({
                                "video_id": video_id,
                                "seed": unigram,
                                "seed_type": "unigram",
                                "hits": hits
                            })
                    except Exception as e:
                        print(f"[TRANSCRIPT-SEED] Error searching unigram '{unigram}': {e}")
                
                # Search for phrases
                for phrase in q_phr:
                    try:
                        cursor.execute("SELECT video_id, text FROM transcripts WHERE text MATCH ?", (f'"{phrase}"',))
                        rows = cursor.fetchall()
                        for video_id, text in rows:
                            hits = text.lower().count(phrase.lower())
                            all_results.append({
                                "video_id": video_id,
                                "seed": phrase,
                                "seed_type": "phrase",
                                "hits": hits
                            })
                    except Exception as e:
                        print(f"[TRANSCRIPT-SEED] Error searching phrase '{phrase}': {e}")
                
                cursor.close()
                conn.close()
            except Exception as e:
                print(f"[TRANSCRIPT-SEED] Error accessing Batch 2 database: {e}")
    
    # Aggregate results by video_id
    video_hits = {}
    for result in all_results:
        video_id = result["video_id"]
        if video_id not in video_hits:
            video_hits[video_id] = {"hits_any": 0, "hits_phr": 0}
        
        if result["seed_type"] == "unigram":
            video_hits[video_id]["hits_any"] += result["hits"]
        else:  # phrase
            video_hits[video_id]["hits_phr"] += result["hits"]
    
    # Create DataFrame
    if not video_hits:
        print("[TRANSCRIPT-SEED] No results found")
        return pd.DataFrame(columns=["video_id", "hits_any", "hits_phr", "text_score", "fused", "t0", "t1"])
    
    rows = []
    for video_id, hits in video_hits.items():
        rows.append({
            "video_id": video_id,
            "hits_any": hits["hits_any"],
            "hits_phr": hits["hits_phr"],
            "t0": 0.0,
            "t1": 30.0
        })
    
    df = pd.DataFrame(rows)
    
    def _minmax(x):
        x = x.astype(float); mn, mx = x.min(), x.max()
        if mx - mn < 1e-9:
            return np.ones_like(x, dtype=float)
        return (x - mn) / (mx - mn)
    
    df["any_n"] = _minmax(df["hits_any"].values)
    df["phr_n"] = _minmax(df["hits_phr"].values)
    a, b = text_weights["any"], text_weights["phr"]
    df["text_score"] = a * df["any_n"] + b * df["phr_n"]
    df["fused"] = df["text_score"]  # Pure text score, no CLIP component
    df = df.sort_values("fused", ascending=False).reset_index(drop=True)
    
    if verbose:
        print(f"[TRANSCRIPT-SEED] fused = {a}*hits_any + {b}*hits_phr | workers={workers} | phrases_only={phrases_only}")
        print(f"[TRANSCRIPT-SEED] Found {len(df)} videos with hits")
    
    if timing:
        print(f"[TRANSCRIPT-SEED] elapsed: {time.perf_counter() - t0:.3f}s (videos={len(df)})")
    
    return df

def rerank_with_transcript_ultra_window(
    clip_results: list,
    query_vi: str,
    seeds: dict,
    top_candidates: int = 60,
    delta_sec: float = 10.0,
    w_clip: float = 0.6, w_text: float = 0.4,
    text_weights: dict | None = None,
    verbose: bool = True, timing: bool = True,
    *,
    workers: int = 8,
    phrases_only: bool = False,
    max_unigrams: int = 16, max_phrases: int = 10,
    num_aspects: int = 1
):
    if text_weights is None:
        text_weights = {"any": 0.3, "phr": 0.7}
    t0 = time.perf_counter()
    cand = [{"video_id": r["video_id"], "clip_total": float(r["total"]),
             "t0": max(0.0, float(r["win"]["t0"]) - float(delta_sec)),
             "t1": float(r["win"]["t1"]) + float(delta_sec)}
            for r in clip_results[:top_candidates]]
    df = pd.DataFrame(cand)
    if df.empty:
        return df
    uni, phr, q_any, q_phr = build_keyword_sets(
        query_vi, seeds,
        max_unigrams=(0 if phrases_only else max_unigrams),
        max_phrases=max_phrases
    )
    # Verbose introspection of what will be used for rerank
    if verbose:
        seed_uni = seeds.get("unigrams", []) if isinstance(seeds, dict) else []
        seed_phr = seeds.get("phrases", []) if isinstance(seeds, dict) else []
        seeds_empty = (len(seed_uni) == 0 and len(seed_phr) == 0)
        # Preview helpers
        def _preview_list(values, limit):
            try:
                values = list(values)
            except Exception:
                values = []
            if len(values) <= limit:
                return values
            return values[:limit] + ["..."]

        def _trim_str(s: str | None, max_len: int = 400) -> str:
            if s is None:
                return "None"
            if len(s) <= max_len:
                return s
            return f"{s[:max_len]}... (+{len(s) - max_len} chars)"

        uni_prev = _preview_list(uni, 24)
        phr_prev = _preview_list(phr, 16)
        print(
            f"[ULTRA-WIN][KEYS] seeds_empty={seeds_empty} | seed_uni={len(seed_uni)} | seed_phr={len(seed_phr)} | "
            f"derived_uni={len(uni)} | derived_phr={len(phr)}"
        )
        print(f"[ULTRA-WIN][UNI] {', '.join(map(str, uni_prev))}")
        print("[ULTRA-WIN][PHR] " + ", ".join(('"{}'.format(p) + '"') if p != "..." else "..." for p in phr_prev))
        print(f"[ULTRA-WIN][q_any] {_trim_str(q_any)}")
        print(f"[ULTRA-WIN][q_phr] {_trim_str(q_phr)}")
    if phrases_only:
        q_any = None
    parts = []
    for _al, db_path, wins in _alias_windows_and_dbpaths(cand):
        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = [ex.submit(_count_one, db_path, w["video_id"], float(w["t0"]), float(w["t1"]), q_any, q_phr)
                       for w in wins]
            for fut in concurrent.futures.as_completed(futures):
                rows.append(fut.result())
        parts.append(pd.DataFrame(rows, columns=["video_id", "hits_any", "hits_phr"]))
    text_df = (pd.concat(parts, ignore_index=True)
               if parts else pd.DataFrame(columns=["video_id", "hits_any", "hits_phr"]))
    def _minmax(x):
        x = x.astype(float); mn, mx = x.min(), x.max()
        if mx - mn < 1e-9:
            return np.ones_like(x, dtype=float)
        return (x - mn) / (mx - mn)
    df = df.merge(text_df, on="video_id", how="left").fillna({"hits_any": 0, "hits_phr": 0})
    df["clip_n"] = _minmax(df["clip_total"].values / num_aspects)
    df["any_n"] = _minmax(df["hits_any"].values)
    df["phr_n"] = _minmax(df["hits_phr"].values)
    a, b = text_weights["any"], text_weights["phr"]
    df["text_score"] = a * df["any_n"] + b * df["phr_n"]
    df["fused"] = w_clip * df["clip_n"] + w_text * df["text_score"]
    df = df.sort_values("fused", ascending=False).reset_index(drop=True)
    if verbose:
        print(f"[ULTRA-WIN] fused = {w_clip}*(clip_total/{num_aspects}) + {w_text}*( {a}*hits_any + {b}*hits_phr ) | Δ={delta_sec}s | workers={workers} | phrases_only={phrases_only}")

    if timing:
        print(f"[ULTRA-WIN] elapsed: {time.perf_counter() - t0:.3f}s (candidates={len(df)})")
    return df

def reorder_results_by_df(orig_results, ranked_df, top_videos=TOP_VIDEOS):
    order = ranked_df.head(top_videos)["video_id"].tolist()
    vid2r = {r["video_id"]: r for r in orig_results}
    return [vid2r[v] for v in order if v in vid2r]

# =============================== MEDIA INFO (watch_url) ==================================
# START: Code to be Replaced (FINAL, COMPLETE, AND FUNCTIONAL VERSION WITH 100% ORIGINAL JS)

def _create_base_player_html(
    video_id: str,
    start_time: float,
    fps: float,
    video_name: str,
    player_html: str,
    player_js_init: str,
    player_js_handlers: str,
) -> str:
    """
    This is the main template for the video player UI.
    It contains ALL the original HTML and JS, with placeholders for player-specific parts.
    """
    vid_us = re.sub(r'[^a-zA-Z0-9_]', '_', video_name)
    start_frame = int(start_time * fps)
    seek_max_init = max(1, start_frame + 1)
    start_frame_plus_1 = start_frame + 1

    # This template now contains the FULL and UNMODIFIED code from your original function.
    html_template = r"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Player</title>
        <<PLAYER_JS_INIT>>
        <style>
            /* This is the full, original CSS block */
            html, body { height: auto !important; min-height: 100% !important; overflow-y: auto !important; }
            body { font-family: Arial, sans-serif; margin: 10px; padding-bottom: 80px; }
            .container, .info-panel, .video-panel { overflow: visible !important; }
            .container { display: flex; gap: 20px; }
            .info-panel { flex: 1; background: #2e2e2e; padding: 20px; border-radius: 8px; color: white; }
            .video-panel { flex: 2; }
            .time-display { font-size: 24px; color: #00ff00; margin: 10px 0; }
            .frame-display { font-size: 20px; color: #ff6b6b; margin: 10px 0; }
            iframe, video { width: 100%; height: 500px; display:block; background-color: #000; }
            .status { margin: 10px 0; }
            .top-form { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 12px 0; }
            .form-group { display: flex; align-items: center; gap: 6px; background: #3a3a3a; padding: 6px 8px; border-radius: 6px; }
            .form-group label { font-size: 12px; color: #ddd; white-space: nowrap; }
            .form-group input, .form-group select { height: 30px; min-width: 110px; padding: 0 8px; border-radius: 6px; border: 1px solid #777; background:#1f1f1f; color:#fff; font-size: 13px; }
            .price-input { width: 100%; display: flex; gap: 6px; margin: 10px 0 8px; align-items: center; }
            .price-input .field { display: flex; width: 100%; height: 36px; align-items: center; }
            .price-input .field span { font-size: 12px; opacity: 0.8; }
            .price-input .field input { width: 100%; height: 100%; outline: none; font-size: 14px; margin-left: 8px; border-radius: 6px; text-align: center; border: 1px solid #777; background:#1f1f1f; color:#fff; min-width: 60px; }
            .price-input .separator { width: 40px; display: flex; font-size: 14px; align-items: center; justify-content: center; }
            .slider { height: 6px; position: relative; background: #444; border-radius: 6px; }
            .slider .progress { height: 100%; left: 0%; right: 100%; position: absolute; border-radius: 6px; background: #17A2B8; }
            .range-input { position: relative; height: 10px; }
            .range-input input[type="range"] { position: absolute; width: 100%; height: 6px; top: -6px; background: none; pointer-events: none; -webkit-appearance: none; -moz-appearance: none; }
            .range-input input[type="range"]::-webkit-slider-thumb { height: 16px; width: 16px; border-radius: 50%; background: #17A2B8; pointer-events: auto; -webkit-appearance: none; box-shadow: 0 0 6px rgba(0,0,0,0.25); border:2px solid #fff; }
            .range-input input[type="range"]::-moz-range-thumb { height: 16px; width: 16px; border:none; border-radius: 50%; background: #17A2B8; pointer-events: auto; -moz-appearance: none; box-shadow: 0 0 6px rgba(0,0,0,0.25); }
            .step-controls { display: flex; align-items: center; gap: 4px; margin-left: 8px; }
            .step-label { font-size: 10px; color: #aaa; white-space: nowrap; }
            .step-input { width: 35px; height: 24px; padding: 2px 4px; border: 1px solid #777; border-radius: 4px; background: #1f1f1f; color: #fff; font-size: 11px; text-align: center; }
            .step-btn { width: 18px; height: 24px; border: 1px solid #777; background: #2a2a2a; color: #fff; border-radius: 4px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 9px; font-weight: bold; }
            .add-btn { width: 80px; height: 32px; border: 1px solid #777; background: #0073e6; color: #fff; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: bold; }
            .current-btn { width: 50px; height: 24px; border: 1px solid #777; background: #28a745; color: #fff; border-radius: 4px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: bold; margin-left: 5px; }
            .preview-wrapper { margin-top: 18px; }
            .preview-card { background:#2e2e2e; border:1px solid #444; border-radius:8px; overflow:hidden; }
            .preview-header { display:flex; align-items:center; justify-content:space-between; padding:10px 12px; background:#1f1f1f; border-bottom:1px solid #444; color:#fff; }
            .preview-controls { display:flex; flex-direction:column; align-items:flex-end; gap:5px; }
            .preview-hint { font-size:12px; opacity:0.7; }
            .delete-range-controls { display:flex; align-items:center; gap:5px; font-size:12px; position: relative; }
            .delete-range-controls input { padding:2px 4px; border:1px solid #777; border-radius:3px; background:#1f1f1f; color:#fff; font-size:11px; }
            .delete-range-btn { padding:2px 8px; border:1px solid #dc3545; background:#dc3545; color:#fff; border-radius:3px; cursor:pointer; font-size:11px; }
            .notification { position: absolute; top: 100%; left: 0; right: 0; background: #28a745; color: white; padding: 8px 12px; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); z-index: 1000; font-size: 12px; font-weight: 500; opacity: 0; transform: translateY(-10px); transition: all 0.3s ease; margin-top: 5px; }
            .notification.show { opacity: 1; transform: translateY(0); }
            .notification.error { background: #dc3545; }
            .trake-confirm-btn { padding: 6px 16px; border: 1px solid #28a745; background: #28a745; color: #fff; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: bold; }
            .save-csv-btn { padding: 8px 20px; border: 1px solid #17a2b8; background: #17a2b8; color: #fff; border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: bold; }
            .trake-scene-input { margin-bottom: 8px; }
            .trake-scene-input label { display: block; font-size: 12px; margin-bottom: 4px; color: #ccc; }
            .trake-scene-input input { width: 200px; padding: 4px 8px; border: 1px solid #777; border-radius: 3px; background: #1f1f1f; color: #fff; font-size: 11px; }
            .trake-scene-btn { width: 30px; height: 30px; border: 2px solid #dc3545; background: #dc3545; color: #fff; border-radius: 50%; cursor: pointer; font-size: 12px; font-weight: bold; display: flex; align-items: center; justify-content: center; transition: all 0.3s ease; }
            .trake-scene-btn.selected { background: #28a745; border-color: #28a745; }
            .pr-list { max-height: 220px; overflow:auto; }
            .pr-row { display:flex; align-items:center; gap:12px; padding:8px 12px; border-bottom:1px dashed #3a3a3a; cursor:pointer; user-select:none; }
            .pr-index { width:34px; text-align:right; color:#ddd; opacity:.6; font-size:12px; }
            .pr-value { color:#fff; font-size:14px; }
            .pr-row.selected { background:#3a3a3a; outline:1px solid #17A2B8; }
            .pr-empty { color:#ccc; opacity:.7; padding:12px; font-size:13px; }
            .sliders-container { margin-top: 20px; padding: 20px; background: #2e2e2e; border-radius: 8px; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="video-panel">
                <<PLAYER_HTML>>
                <div class="sliders-container">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                        <div id="trake-scene-number-wrap-<<vid_us>>" style="display:none;">
                            <label style="font-size:12px; color:#ccc; margin-right:8px;">Add for:</label>
                            <div id="trake-scene-buttons-<<vid_us>>" style="display:inline-flex; gap:5px; margin-left:8px;"></div>
                        </div>
                        <div style="display:flex; align-items:center; gap:8px; margin-left:auto;">
                            <label for="goto-frame-<<vid_us>>" style="font-size:12px; color:#ccc;"><strong>Go to Frame:</strong></label>
                            <input type="number" id="goto-frame-<<vid_us>>" placeholder="Enter frame and press Enter" style="width: 180px; height:28px; padding: 2px 6px; border-radius: 3px; border: 1px solid #777; background:#1f1f1f; color:#fff; font-size:11px;" />
                        </div>
                    </div>
                    <div style="margin-top:10px;">
                        <div style="display:flex; justify-content:space-between;">
                            <span id="frame-left-<<vid_us>>"><strong><<start_frame>> F</strong></span>
                            <span id="frame-right-<<vid_us>>"><strong>-- F</strong></span>
                        </div>
                        <input type="range" id="seek-frame-<<vid_us>>" min="0" max="<<seek_max_init>>" step="1" value="<<start_frame>>" style="width:100%;">
                        <div id="add-single-wrap-<<vid_us>>" style="display:none; margin-top:10px; text-align:center;">
                            <button id="add-single-<<vid_us>>" class="add-btn">Add</button>
                        </div>
                        <div id="trake-add-single-wrap-<<vid_us>>" style="display:none; margin-top:10px; text-align:center;">
                            <button id="trake-add-single-<<vid_us>>" class="add-btn">Add</button>
                        </div>
                        <div style="margin-top:10px;">
                            <div class="price-input" id="rs-inputs-<<vid_us>>">
                                <div class="field" style="flex: 1;">
                                    <span>Min</span>
                                    <input type="number" class="input-min" id="rs-input-min-<<vid_us>>" value="<<start_frame>>">
                                    <button class="current-btn" id="rs-min-current-<<vid_us>>" title="Set Min to Current Frame">Current</button>
                                </div>
                                <div class="separator">-</div>
                                <div class="field" style="flex: 1;">
                                    <span>Max</span>
                                    <input type="number" class="input-max" id="rs-input-max-<<vid_us>>" value="<<start_frame_plus_1>>">
                                    <button class="current-btn" id="rs-max-current-<<vid_us>>" title="Set Max to Current Frame">Current</button>
                                </div>
                                <div class="step-controls">
                                    <span class="step-label">Step:</span>
                                    <input type="number" class="step-input" id="rs-step-size-<<vid_us>>" value="5" min="1" max="100">
                                    <button class="step-btn" id="rs-max-up-<<vid_us>>" title="Tăng Max">▲</button>
                                    <button class="step-btn" id="rs-max-down-<<vid_us>>" title="Giảm Max">▼</button>
                                </div>
                            </div>
                            <div class="slider" id="rs-slider-<<vid_us>>">
                                <div class="progress" id="rs-progress-<<vid_us>>"></div>
                            </div>
                            <div class="range-input">
                                <input type="range" class="range-min" id="rs-range-min-<<vid_us>>" min="0" max="<<seek_max_init>>" value="<<start_frame>>" step="1">
                                <input type="range" class="range-max" id="rs-range-max-<<vid_us>>" min="0" max="<<seek_max_init>>" value="<<start_frame_plus_1>>" step="1">
                            </div>
                            <div id="add-batch-wrap-<<vid_us>>" style="display:none; margin-top:10px; text-align:center;">
                                <div id="frames-total-display-<<vid_us>>" style="margin-bottom:8px; font-size:12px; color:#ccc;">
                                    <strong>Frames total: <span id="frames-total-count-<<vid_us>>">0</span></strong>
                                </div>
                                <button id="add-batch-<<vid_us>>" class="add-btn">Add Batch</button>
                            </div>
                            <div id="trake-add-batch-wrap-<<vid_us>>" style="display:none; margin-top:10px; text-align:center;">
                                <div id="trake-frames-total-display-<<vid_us>>" style="margin-bottom:8px; font-size:12px; color:#ccc;">
                                    <strong>Frames total: <span id="trake-frames-total-count-<<vid_us>>">0</span></strong>
                                </div>
                                <button id="trake-add-batch-<<vid_us>>" class="add-btn">Add Batch</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="info-panel">
                <div class="time-display" id="current-time-<<vid_us>>">00:00:00</div>
                <h4>Current Frame:</h4>
                <div class="frame-display" id="current-frame-<<vid_us>>"><<start_frame>></div>
                <div class="status">
                     <p><strong>Start Time:</strong> <span id="start-time-display-<<vid_us>>">00:00</span></p>
                    <p><strong>Start Frame:</strong> <<start_frame>></p>
                    <p><strong>FPS:</strong> <<fps>></p>
                    <p><strong>Status:</strong> <span id="status-<<vid_us>>">Initializing...</span></p>
                    <p><strong>Player State:</strong> <span id="player-state-<<vid_us>>">Unknown</span></p>
                    <div class="top-form" id="qa-form-<<vid_us>>">
                        <div class="form-group">
                            <label for="question-count-<<vid_us>>">Câu hỏi:</label>
                            <input type="number" id="question-count-<<vid_us>>" min="1" placeholder="Nhập số câu hỏi">
                        </div>
                        <div class="form-group">
                            <label for="task-select-<<vid_us>>">Task:</label>
                            <select id="task-select-<<vid_us>>">
                                <option value="KIS">KIS</option>
                                <option value="QA">QA</option>
                                <option value="TRAKE">TRAKE</option>
                            </select>
                        </div>
                        <div class="form-group" id="qa-answer-wrap-<<vid_us>>" style="display:none;">
                            <label for="qa-answer-<<vid_us>>">Đáp án QA:</label>
                            <input type="text" id="qa-answer-<<vid_us>>" placeholder="Nhập đáp án">
                        </div>
                        <div class="form-group" id="trake-scenes-wrap-<<vid_us>>" style="display:none;">
                            <label for="trake-scenes-<<vid_us>>">Scenes:</label>
                              <input type="number" id="trake-scenes-<<vid_us>>" min="1" max="10" placeholder="Nhập số cảnh" style="width:40px; height:40px; text-align:center;">
                              <div id="trake-scene-inputs-<<vid_us>>" style="margin-top:10px;"></div>
                              <div id="trake-total-display-<<vid_us>>" style="margin-top:10px; padding:8px; background:#333; border-radius:4px; text-align:center; color:#fff; font-size:12px; display:none;">
                                  <strong>Tổng số frame: <span id="trake-total-count-<<vid_us>>">0</span></strong>
                              </div>
                              <div style="text-align:right; margin-top:20px;">
                                  <button id="trake-confirm-<<vid_us>>" class="trake-confirm-btn" style="display:none;">Confirm</button>
                                <div id="trake-success-notification-<<vid_us>>" class="notification" style="position:relative; top:auto; left:auto; right:auto; margin-top:10px; text-align:center;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div style="text-align:right; margin:1px 0; padding:1px 0; position:relative;">
            <button id="import-csv-<<vid_us>>" class="save-csv-btn import-btn" style="margin-right:10px; background:#28a745; border-color:#28a745;">Import CSV</button>
            <button id="export-csv-<<vid_us>>" class="save-csv-btn">Export CSV</button>
            <input type="file" id="csv-file-input-<<vid_us>>" accept=".csv" style="display:none;">
            <div id="csv-notification-<<vid_us>>" class="notification" style="position:absolute; top:100%; right:0; left:auto; margin-top:5px; text-align:right; max-width:300px;"></div>
        </div>
        <div class="preview-wrapper">
            <div class="preview-card" id="preview-card-<<vid_us>>">
                <div class="preview-header">
                    <div><strong>Preview Result</strong></div>
                    <div class="preview-controls">
                        <div class="preview-hint">Chọn dòng rồi nhấn <kbd>Delete</kbd> để xóa</div>
                        <div class="delete-range-controls">
                            <span>Delete range:</span>
                            <input type="number" id="delete-from-<<vid_us>>" placeholder="From" min="1" style="width:60px; margin:0 5px;">
                            <span>-</span>
                            <input type="number" id="delete-to-<<vid_us>>" placeholder="To" min="1" style="width:60px; margin:0 5px;">
                            <button id="delete-range-btn-<<vid_us>>" class="delete-range-btn">Delete</button>
                            <div id="notification-<<vid_us>>" class="notification"></div>
                        </div>
                    </div>
                </div>
                <div class="pr-list" id="pr-list-<<vid_us>>"></div>
                <div class="pr-empty" id="pr-empty-<<vid_us>>" style="display:none;">(Trống)</div>
            </div>
        </div>

        <script>
            var timeUpdateInterval_<<vid_us>>;
            var fps = <<fps>>;
            var startTime = <<start_time>>;
            var previewData_<<vid_us>> = [];
            window._seeking_<<vid_us>> = false;

            (function ensureScrollable(){ try { document.documentElement.style.overflowY = 'auto'; document.body.style.overflowY = 'auto'; } catch(_){} })();

            function formatTimeMMSS(s){var m=Math.floor(s/60),c=Math.floor(s%60);return`${String(m).padStart(2,'0')}:${String(c).padStart(2,'0')}`}
            function formatTimeMMSSFF(s,fps){var m=Math.floor(s/60),c=Math.floor(s%60),f=Math.floor((s%1)*fps);return`${String(m).padStart(2,'0')}:${String(c).padStart(2,'0')}:${String(f).padStart(2,'0')}`}
            function calculateFrame(t){return Math.floor(t*fps);}

            function renderPreview_<<vid_us>>() {
                var list = document.getElementById('pr-list-<<vid_us>>'); if (!list) return;
                list.innerHTML = '';
                previewData_<<vid_us>>.forEach(function(val, idx){
                    var row = document.createElement('div'); row.className = 'pr-row'; row.tabIndex = 0; row.dataset.index = idx;
                    row.innerHTML = '<div class="pr-index">'+String(idx+1).padStart(2,"0")+'</div><div class="pr-value">'+val+'</div>';
                    row.addEventListener('click', function(e){ if (!e.ctrlKey && !e.metaKey) { document.querySelectorAll('#pr-list-<<vid_us>> .pr-row.selected').forEach(function(r){ r.classList.remove('selected'); }); } row.classList.toggle('selected'); row.focus(); });
                    list.appendChild(row);
                });
                var empty = document.getElementById('pr-empty-<<vid_us>>'); if (empty) empty.style.display = previewData_<<vid_us>>.length ? 'none' : 'block';
            }
            function showNotification_<<vid_us>>(message, isError=false) { var n=document.getElementById('notification-<<vid_us>>'); if(!n)return; n.textContent=message; n.className='notification'+(isError?' error':''); n.classList.add('show'); setTimeout(function(){n.classList.remove('show');},3000); }
            function isInputLike_<<vid_us>>(el){ if (!el) return false; var tag=(el.tagName||'').toUpperCase(); return el.isContentEditable || tag==='INPUT' || tag==='TEXTAREA' || tag==='SELECT'; }
            function deleteSelectedPreviewRows_<<vid_us>>() {
                var list = document.getElementById('pr-list-<<vid_us>>'); if (!list) return;
                var selected = Array.from(list.querySelectorAll('.pr-row.selected')); if (!selected.length) { showNotification_<<vid_us>>('Vui lòng chọn dòng cần xóa', true); return; }
                var idxs = selected.map(function(el){ return parseInt(el.dataset.index); }).sort(function(a,b){ return b-a; });
                idxs.forEach(function(i){ if (i>=0 && i<previewData_<<vid_us>>.length) previewData_<<vid_us>>.splice(i,1); });
                renderPreview_<<vid_us>>(); showNotification_<<vid_us>>('Đã xóa ' + idxs.length + ' dòng được chọn');
            }
            document.addEventListener('keydown', function(e){ if (e.key==='Delete' && !isInputLike_<<vid_us>>(document.activeElement)) { if (document.querySelector('#pr-list-<<vid_us>> .pr-row.selected')) { e.preventDefault(); deleteSelectedPreviewRows_<<vid_us>>(); } } });

            // --- Player-specific JS handlers will be injected here ---
            <<PLAYER_JS_HANDLERS>>

            // --- THIS IS THE FULL, RESTORED DOMContentLoaded LISTENER FROM YOUR ORIGINAL FILE ---
            document.addEventListener('DOMContentLoaded', function(){
                renderPreview_<<vid_us>>();
                var startTimeEl = document.getElementById('start-time-display-<<vid_us>>');
                if (startTimeEl) startTimeEl.textContent = formatTimeMMSS(startTime);

                var gotoInput = document.getElementById('goto-frame-<<vid_us>>');
                if (gotoInput){ gotoInput.addEventListener('keydown',function(e){ if(e.key==='Enter'){ var f=parseInt(gotoInput.value,10); if(!isNaN(f)&&f>=0){ var t=f/fps; try{ player_<<vid_us>>.seekTo(t,true); } catch(_){ try { player_<<vid_us>>.currentTime = t; } catch(__){} } } } }); }
                
                var taskSel=document.getElementById('task-select-<<vid_us>>'), qaWrap=document.getElementById('qa-answer-wrap-<<vid_us>>'), scenesWrap=document.getElementById('trake-scenes-wrap-<<vid_us>>'), addSingleWrap=document.getElementById('add-single-wrap-<<vid_us>>'), addBatchWrap=document.getElementById('add-batch-wrap-<<vid_us>>'), framesTotalDisplay=document.getElementById('frames-total-display-<<vid_us>>'), framesTotalCount=document.getElementById('frames-total-count-<<vid_us>>'), trakeAddSingleWrap=document.getElementById('trake-add-single-wrap-<<vid_us>>'), trakeAddBatchWrap=document.getElementById('trake-add-batch-wrap-<<vid_us>>'), trakeFramesTotalDisplay=document.getElementById('trake-frames-total-display-<<vid_us>>'), trakeFramesTotalCount=document.getElementById('trake-frames-total-count-<<vid_us>>'), trakeSceneNumberWrap=document.getElementById('trake-scene-number-wrap-<<vid_us>>'), trakeSceneButtons=document.getElementById('trake-scene-buttons-<<vid_us>>'), trakeSceneInputs=document.getElementById('trake-scene-inputs-<<vid_us>>'), trakeConfirmBtn=document.getElementById('trake-confirm-<<vid_us>>'), trakeScenesInput=document.getElementById('trake-scenes-<<vid_us>>'), trakeTotalDisplay=document.getElementById('trake-total-display-<<vid_us>>'), trakeTotalCount=document.getElementById('trake-total-count-<<vid_us>>');
                var selectedSceneNum = 1;

                function updateTaskUI(){ if(!taskSel)return; var v=taskSel.value; if(qaWrap)qaWrap.style.display=v==='QA'?'flex':'none'; if(scenesWrap)scenesWrap.style.display=v==='TRAKE'?'flex':'none'; var k=v==='KIS'||v==='QA'; if(addSingleWrap)addSingleWrap.style.display=k?'block':'none'; if(addBatchWrap)addBatchWrap.style.display=k?'block':'none'; if(framesTotalDisplay)framesTotalDisplay.style.display=k?'block':'none'; var t=v==='TRAKE'; if(trakeAddSingleWrap)trakeAddSingleWrap.style.display=t?'block':'none'; if(trakeAddBatchWrap)trakeAddBatchWrap.style.display=t?'block':'none'; if(trakeFramesTotalDisplay)trakeFramesTotalDisplay.style.display=t?'block':'none'; if(trakeSceneNumberWrap)trakeSceneNumberWrap.style.display=t?'block':'none'; if(trakeTotalDisplay)trakeTotalDisplay.style.display=t?'block':'none'; }
                if(taskSel){ taskSel.addEventListener('change', updateTaskUI); updateTaskUI(); }

                function getCurrentFrame() { try { return calculateFrame(player_<<vid_us>>.getCurrentTime()); } catch(e) { try { return calculateFrame(player_<<vid_us>>.currentTime); } catch(e2) { return 0; } } }
                var addSingleBtn=document.getElementById('add-single-<<vid_us>>'), qaAnswerInput=document.getElementById('qa-answer-<<vid_us>>');
                if(addSingleBtn){ addSingleBtn.addEventListener('click', function(){ if(previewData_<<vid_us>>.length>=100){alert('Limit 100 rows'); return;} var cf=getCurrentFrame(); var vn='<<video_name>>'; var task=taskSel.value; if(task==='KIS'){previewData_<<vid_us>>.push(vn+', '+cf);}else if(task==='QA'){var a=qaAnswerInput.value.trim();if(!a){alert('Please enter QA answer');return;}previewData_<<vid_us>>.push(vn+', '+cf+', "'+a+'"');} renderPreview_<<vid_us>>(); }); }
                var addBatchBtn=document.getElementById('add-batch-<<vid_us>>');
                if(addBatchBtn){ addBatchBtn.addEventListener('click', function(){ var minF=parseInt(document.getElementById('rs-input-min-<<vid_us>>').value)||0, maxF=parseInt(document.getElementById('rs-input-max-<<vid_us>>').value)||0; var vn='<<video_name>>'; var task=taskSel.value; var frames=[]; var a=qaAnswerInput.value.trim(); if(task==='QA'&&!a){alert('Please enter QA answer');return;} for(var f=minF;f<=maxF;f++){if(previewData_<<vid_us>>.length+frames.length>=100)break; if(task==='KIS')frames.push(vn+', '+f); else if(task==='QA')frames.push(vn+', '+f+', "'+a+'"');} previewData_<<vid_us>>.push(...frames); renderPreview_<<vid_us>>(); }); }

                var frameSlider=document.getElementById('seek-frame-<<vid_us>>'), frameLeft=document.getElementById('frame-left-<<vid_us>>');
                if(frameSlider){
                    frameSlider.addEventListener('input',function(){ window._seeking_<<vid_us>>=true; var f=parseInt(frameSlider.value,10)||0; if(frameLeft) frameLeft.innerHTML='<strong>'+f+' F</strong>'; });
                    frameSlider.addEventListener('change',function(){ var f=parseInt(frameSlider.value,10)||0; var t=f/fps; try{player_<<vid_us>>.seekTo(t,true);}catch(_){try{player_<<vid_us>>.currentTime=t;}catch(__){}} window._seeking_<<vid_us>>=false; });
                    frameSlider.addEventListener('mousedown',function(){ window._seeking_<<vid_us>>=true; });
                    frameSlider.addEventListener('mouseup',function(){ window._seeking_<<vid_us>>=false; });
                }

                var rangeMin=document.getElementById('rs-range-min-<<vid_us>>'), rangeMax=document.getElementById('rs-range-max-<<vid_us>>'), inputMin=document.getElementById('rs-input-min-<<vid_us>>'), inputMax=document.getElementById('rs-input-max-<<vid_us>>'), progress=document.getElementById('rs-progress-<<vid_us>>');
                function updateRangeSlider(){ var mn=parseInt(rangeMin.value), mx=parseInt(rangeMax.value); if(mx<mn){mx=mn;rangeMax.value=mn;} if(mn>mx){mn=mx;rangeMin.value=mx;} inputMin.value=mn; inputMax.value=mx; var maxV=parseInt(rangeMin.max); if(progress){progress.style.left=(mn/maxV*100)+'%'; progress.style.right=(100-(mx/maxV*100))+'%';} var total=mx-mn+1; if(framesTotalCount)framesTotalCount.textContent=total; if(trakeFramesTotalCount)trakeFramesTotalCount.textContent=total;}
                if(rangeMin)rangeMin.addEventListener('input', updateRangeSlider);
                if(rangeMax)rangeMax.addEventListener('input', updateRangeSlider);
                var minCurrentBtn=document.getElementById('rs-min-current-<<vid_us>>'); if(minCurrentBtn)minCurrentBtn.addEventListener('click',function(){rangeMin.value=getCurrentFrame(); updateRangeSlider();});
                var maxCurrentBtn=document.getElementById('rs-max-current-<<vid_us>>'); if(maxCurrentBtn)maxCurrentBtn.addEventListener('click',function(){rangeMax.value=getCurrentFrame(); updateRangeSlider();});
                var stepInput=document.getElementById('rs-step-size-<<vid_us>>'), maxUp=document.getElementById('rs-max-up-<<vid_us>>'), maxDown=document.getElementById('rs-max-down-<<vid_us>>');
                if(maxUp)maxUp.addEventListener('click',function(){var s=parseInt(stepInput.value); rangeMax.value=Math.min(parseInt(rangeMax.max), parseInt(rangeMax.value)+s); updateRangeSlider();});
                if(maxDown)maxDown.addEventListener('click',function(){var s=parseInt(stepInput.value); rangeMax.value=Math.max(parseInt(rangeMax.min), parseInt(rangeMax.value)-s); updateRangeSlider();});
                
                var delRangeBtn=document.getElementById('delete-range-btn-<<vid_us>>');
                if(delRangeBtn)delRangeBtn.addEventListener('click', function(){
                     var fromInput = document.getElementById('delete-from-<<vid_us>>'); var toInput = document.getElementById('delete-to-<<vid_us>>'); if (!fromInput || !toInput) return;
                     var from=parseInt(fromInput.value), to=parseInt(toInput.value);
                     if(isNaN(from)||isNaN(to)||from<1||to<from||to>previewData_<<vid_us>>.length){ showNotification_<<vid_us>>('Invalid range',true); return; }
                     previewData_<<vid_us>>.splice(from-1, to-from+1); fromInput.value=''; toInput.value=''; renderPreview_<<vid_us>>(); showNotification_<<vid_us>>('Deleted rows '+from+'-'+to);
                });

                var exportCsvBtn=document.getElementById('export-csv-<<vid_us>>'), importCsvBtn=document.getElementById('import-csv-<<vid_us>>'), csvFileInput=document.getElementById('csv-file-input-<<vid_us>>'), csvNotification=document.getElementById('csv-notification-<<vid_us>>');
                function showCsvNotification(message,isError=false){ if(!csvNotification)return; csvNotification.textContent=message; if(isError){csvNotification.classList.add('error');}else{csvNotification.classList.remove('error');} csvNotification.classList.add('show'); setTimeout(function(){csvNotification.classList.remove('show');},3000); }
                if(exportCsvBtn){ exportCsvBtn.addEventListener('click', function(){ try { var qCount=document.getElementById('question-count-<<vid_us>>').value; if(!qCount||parseInt(qCount)<=0){showCsvNotification('Please enter question number',true); return;} if(!previewData_<<vid_us>>||previewData_<<vid_us>>.length===0){showCsvNotification('No data to export',true); return;} var task=document.getElementById('task-select-<<vid_us>>').value.toLowerCase(); var filename='query-'+qCount+'-'+task+'.csv'; var csv=previewData_<<vid_us>>.join('\\n'); var blob=new Blob([csv],{type:'text/csv;charset=utf-8;'}); var url=window.URL.createObjectURL(blob); var a=document.createElement('a'); a.href=url; a.download=filename; document.body.appendChild(a); a.click(); document.body.removeChild(a); window.URL.revokeObjectURL(url); showCsvNotification('Exported '+filename); } catch(e){ showCsvNotification('Export failed: '+e.message,true); } }); }
                if(importCsvBtn&&csvFileInput){ importCsvBtn.addEventListener('click',function(){ if(previewData_<<vid_us>>&&previewData_<<vid_us>>.length>0){showCsvNotification('Can only import to empty preview.',true);return;} csvFileInput.click(); }); csvFileInput.addEventListener('change',function(event){ var file=event.target.files[0]; if(!file)return; var reader=new FileReader(); reader.onload=function(e){ try{var content=e.target.result; var lines=content.split(/\\r\\n|\\n/).filter(function(line){return line.trim()!==''}); if(lines.length===0){showCsvNotification('Empty or invalid CSV file.',true);return;} previewData_<<vid_us>>=lines; renderPreview_<<vid_us>>(); showCsvNotification('Imported '+lines.length+' rows from '+file.name, false);}catch(err){showCsvNotification('Error reading file: '+err.message,true);} }; reader.readAsText(file); event.target.value=''; }); }

                // TRAKE Logic
                if(trakeScenesInput)trakeScenesInput.addEventListener('input',function(){ var n=parseInt(this.value); if(n>0&&n<=10){createTrakeSceneInputs(n);createTrakeSceneButtons(n);}else{if(trakeSceneInputs)trakeSceneInputs.innerHTML='';if(trakeSceneButtons)trakeSceneButtons.innerHTML='';if(trakeConfirmBtn)trakeConfirmBtn.style.display='none';} });
                function createTrakeSceneButtons(n){ if(!trakeSceneButtons)return; trakeSceneButtons.innerHTML=''; for(var i=1;i<=n;i++){var b=document.createElement('button'); b.className='trake-scene-btn'; b.id='trake-scene-btn-'+i+'-<<vid_us>>'; b.textContent=i; b.addEventListener('click',function(i){return function(){trakeSceneButtons.querySelectorAll('.selected').forEach(el=>el.classList.remove('selected'));this.classList.add('selected');selectedSceneNum=i;};}(i)); trakeSceneButtons.appendChild(b);} if(n>0){trakeSceneButtons.firstChild.classList.add('selected'); selectedSceneNum=1;} }
                function createTrakeSceneInputs(n){ if(!trakeSceneInputs)return; trakeSceneInputs.innerHTML=''; for(var i=1;i<=n;i++){var d=document.createElement('div'); d.className='trake-scene-input'; d.innerHTML='<label>Scene '+i+': Total <span id="trake-scene-count-'+i+'-<<vid_us>>" style="color:#4CAF50;">0</span> frames</label><input type="text" id="trake-scene-'+i+'-<<vid_us>>" placeholder="Frame or [start,end]">'; trakeSceneInputs.appendChild(d); d.querySelector('input').addEventListener('input',calculateTrakeTotal);} if(trakeConfirmBtn)trakeConfirmBtn.style.display='block'; if(trakeTotalDisplay)trakeTotalDisplay.style.display='block'; calculateTrakeTotal(); }
                function calculateTrakeTotal(){ if(!trakeScenesInput||!trakeTotalCount)return; var numScenes=parseInt(trakeScenesInput.value)||0; if(numScenes===0){trakeTotalCount.textContent='0';return;} var total=1, hasInput=false; for(var i=1;i<=numScenes;i++){var inp=document.getElementById('trake-scene-'+i+'-<<vid_us>>'); var countDisp=document.getElementById('trake-scene-count-'+i+'-<<vid_us>>'); if(!inp)continue; var v=inp.value.trim(), sc=0; if(v.startsWith('[')&&v.endsWith(']')){var p=v.slice(1,-1).split(','); if(p.length===2){var s=parseInt(p[0].trim()), e=parseInt(p[1].trim()); if(!isNaN(s)&&!isNaN(e)&&s<=e)sc=e-s+1;}}else{if(!isNaN(parseInt(v)))sc=1;} if(countDisp)countDisp.textContent=sc; if(sc>0){total*=sc; hasInput=true;}} trakeTotalCount.textContent=hasInput?total.toLocaleString():'0'; }
                if(trakeConfirmBtn)trakeConfirmBtn.addEventListener('click', function(){ generateTrakeCombinations(); });
                function generateTrakeCombinations() {
                    var combinations = []; var sceneInputs = trakeSceneInputs.querySelectorAll('.trake-scene-input input'); var videoName = '<<video_name>>';
                    var sceneData = [];
                    for (var i = 0; i < sceneInputs.length; i++) {
                        var value = sceneInputs[i].value.trim(); if (!value) { showTrakeSuccessNotification('Vui lòng nhập đầy đủ tất cả các scene', true); return; }
                        var frames = [];
                        if (value.startsWith('[') && value.endsWith(']')) { var range = value.slice(1, -1).split(','); if (range.length !== 2) { showTrakeSuccessNotification('Format range không đúng. Sử dụng [start,end]', true); return; } var start = parseInt(range[0]); var end = parseInt(range[1]); if (isNaN(start) || isNaN(end) || start > end) { showTrakeSuccessNotification('Range không hợp lệ', true); return; } for (var frame = start; frame <= end; frame++) { frames.push(frame); } } else { var frame = parseInt(value); if (isNaN(frame)) { showTrakeSuccessNotification('Frame không hợp lệ: ' + value, true); return; } frames.push(frame); }
                        sceneData.push(frames);
                    }
                    function generateCombos(scenes, currentCombo, index) { if (index === scenes.length) { combinations.push([videoName].concat(currentCombo)); return; } for (var i = 0; i < scenes[index].length; i++) { generateCombos(scenes, currentCombo.concat(scenes[index][i]), index + 1); } }
                    generateCombos(sceneData, [], 0);
                    if (previewData_<<vid_us>>.length + combinations.length > 100) { showTrakeSuccessNotification('Sẽ vượt quá giới hạn 100 dòng! Chỉ thêm được ' + (100 - previewData_<<vid_us>>.length) + ' combinations.', true); return; }
                    var formattedCombinations = combinations.map(function(combo) { return combo.join(', '); });
                    previewData_<<vid_us>>.push.apply(previewData_<<vid_us>>, formattedCombinations); renderPreview_<<vid_us>>();
                    var notif = document.getElementById('trake-success-notification-<<vid_us>>'); if(notif){notif.textContent = 'Đã thêm ' + combinations.length + ' combinations thành công!'; notif.classList.remove('error'); notif.classList.add('show'); setTimeout(()=>notif.classList.remove('show'), 3000);}
                }
                var trakeAddSingleBtn=document.getElementById('trake-add-single-<<vid_us>>'); if(trakeAddSingleBtn)trakeAddSingleBtn.addEventListener('click',function(){ var cf=getCurrentFrame(); var inp=document.getElementById('trake-scene-'+selectedSceneNum+'-<<vid_us>>'); if(inp){inp.value=cf; var btn=document.getElementById('trake-scene-btn-'+selectedSceneNum+'-<<vid_us>>'); if(btn)btn.classList.add('selected'); calculateTrakeTotal(); autoAdvanceToNextScene();} });
                var trakeAddBatchBtn=document.getElementById('trake-add-batch-<<vid_us>>'); if(trakeAddBatchBtn)trakeAddBatchBtn.addEventListener('click',function(){ var minF=parseInt(document.getElementById('rs-input-min-<<vid_us>>').value)||0; var maxF=parseInt(document.getElementById('rs-input-max-<<vid_us>>').value)||0; var inp=document.getElementById('trake-scene-'+selectedSceneNum+'-<<vid_us>>'); if(inp){inp.value='['+minF+','+maxF+']'; var btn=document.getElementById('trake-scene-btn-'+selectedSceneNum+'-<<vid_us>>'); if(btn)btn.classList.add('selected'); calculateTrakeTotal(); autoAdvanceToNextScene();} });
                function autoAdvanceToNextScene(){ var nextNum=selectedSceneNum+1; if(nextNum<=trakeSceneButtons.children.length){ document.getElementById('trake-scene-btn-'+selectedSceneNum+'-<<vid_us>>').classList.remove('selected'); selectedSceneNum=nextNum; document.getElementById('trake-scene-btn-'+selectedSceneNum+'-<<vid_us>>').classList.add('selected'); } }
            });
        </script>
    </body>
    </html>
    """
    # Replace all placeholders
    full_html = (html_template
        .replace("<<vid_us>>", vid_us)
        .replace("<<video_name>>", video_name)
        .replace("<<start_time>>", f"{start_time:.1f}")
        .replace("<<fps>>", f"{fps}")
        .replace("<<start_frame>>", str(start_frame))
        .replace("<<start_frame_plus_1>>", str(start_frame_plus_1))
        .replace("<<seek_max_init>>", str(seek_max_init))
        .replace("<<PLAYER_HTML>>", player_html)
        .replace("<<PLAYER_JS_INIT>>", player_js_init)
        .replace("<<PLAYER_JS_HANDLERS>>", player_js_handlers)
    )
    return full_html


def create_youtube_player_html(youtube_id: str, start_time: float, fps: float, video_name: str) -> str:
    vid_us = re.sub(r'[^a-zA-Z0-9_]', '_', video_name)
    player_html = f'<iframe id="ytplayer-{vid_us}" src="https://www.youtube.com/embed/{youtube_id}?start={int(start_time)}&enablejsapi=1&autoplay=0&rel=0&modestbranding=1&showinfo=0&controls=1&fs=1&cc_load_policy=0&iv_load_policy=3&playsinline=1" frameborder="0" allowfullscreen></iframe>'
    # The original code dynamically created the script tag. This is a robust pattern.
    player_js_init = """
        <script>
            if (!window.YT) { // Only create script tag if not already there
                var tag = document.createElement('script');
                tag.src = "https://www.youtube.com/iframe_api";
                var firstScriptTag = document.getElementsByTagName('script')[0];
                firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
            }
        </script>
    """
    player_js_handlers = f"""
    var player_{vid_us};
    function onYouTubeIframeAPIReady(){{ 
        console.log('[DEBUG] Creating YouTube player for: {vid_us}');
        if(!document.getElementById('ytplayer-{vid_us}')) {{
            console.log('[DEBUG] Element ytplayer-{vid_us} not found, skipping');
            return; 
        }}
        // Stop all other YouTube players before creating new one
        if(window.stopAllYouTubePlayers) {{
            console.log('[DEBUG] Stopping all other YouTube players before creating {vid_us}');
            window.stopAllYouTubePlayers();
        }}
        player_{vid_us}=new YT.Player('ytplayer-{vid_us}',{{events:{{'onReady':onPlayerReady_{vid_us},'onStateChange':onPlayerStateChange_{vid_us}}}}}); 
        // Register this player for global management
        if(!window.youtubePlayers) window.youtubePlayers = [];
        window.youtubePlayers.push(player_{vid_us});
        console.log('[DEBUG] Registered player {vid_us}, total players:', window.youtubePlayers.length);
    }}
    function onPlayerReady_{vid_us}(event){{
        console.log('[DEBUG] Player {vid_us} is ready');
        document.getElementById('status-{vid_us}').textContent='Player Ready';
        try{{
            // Force video to load and display properly
            event.target.seekTo({start_time}, true);
            // Try to play and immediately pause to trigger video load
            event.target.playVideo();
            setTimeout(function() {{
                try {{ event.target.pauseVideo(); }} catch(e) {{ console.log('[DEBUG] Error pausing after load:', e); }}
            }}, 100);
        }}catch(e){{
            console.log('[DEBUG] Error in onPlayerReady:', e);
        }}
        try{{var d=event.target.getDuration(); if(d){{var mF=Math.floor(d*{fps});document.getElementById('seek-frame-{vid_us}').max=mF;document.getElementById('rs-range-min-{vid_us}').max=mF;document.getElementById('rs-range-max-{vid_us}').max=mF;document.getElementById('frame-right-{vid_us}').innerHTML='<strong>'+mF+' F</strong>';}} }}catch(_){{}}
        if(timeUpdateInterval_{vid_us})clearInterval(timeUpdateInterval_{vid_us}); timeUpdateInterval_{vid_us}=setInterval(updateTimeAndFrame_{vid_us},200);
    }}
    function onPlayerStateChange_{vid_us}(e){{ 
        var s={{0:'Ended',1:'Playing',2:'Paused',3:'Buffering',5:'Cued', '-1': 'Unstarted'}}; 
        console.log('[DEBUG] Player {vid_us} state changed to:', s[e.data] || 'Unknown', '(', e.data, ')');
        document.getElementById('player-state-{vid_us}').textContent=s[e.data]||'Unknown'; 
        updateTimeAndFrame_{vid_us}();
        // If this player starts playing, stop all other players
        if(e.data === 1 && window.stopAllYouTubePlayers) {{
            console.log('[DEBUG] Player {vid_us} started playing, stopping all other players');
            window.stopAllYouTubePlayers(player_{vid_us});
        }}
    }}
    function updateTimeAndFrame_{vid_us}(){{
        if(!player_{vid_us}||typeof player_{vid_us}.getCurrentTime !== 'function')return;
        var t=player_{vid_us}.getCurrentTime(), f=calculateFrame(t);
        document.getElementById('current-time-{vid_us}').textContent=formatTimeMMSSFF(t,{fps});
        document.getElementById('current-frame-{vid_us}').textContent=f;
        var s=document.getElementById('seek-frame-{vid_us}'); if(s&&!window._seeking_{vid_us}){{s.value=f;document.getElementById('frame-left-{vid_us}').innerHTML='<strong>'+f+' F</strong>';}}
    }}
    // Global function to stop all YouTube players except the current one
    if(!window.stopAllYouTubePlayers) {{
        window.stopAllYouTubePlayers = function(currentPlayer) {{
            console.log('[DEBUG] stopAllYouTubePlayers called, currentPlayer:', currentPlayer ? 'provided' : 'none');
            if(window.youtubePlayers) {{
                console.log('[DEBUG] Found', window.youtubePlayers.length, 'registered players');
                window.youtubePlayers.forEach(function(player, index) {{
                    if(player && player !== currentPlayer && typeof player.pauseVideo === 'function') {{
                        try {{ 
                            console.log('[DEBUG] Pausing player', index);
                            player.pauseVideo(); 
                        }} catch(e) {{ 
                            console.log('[DEBUG] Error pausing player', index, ':', e); 
                        }}
                    }}
                }});
            }} else {{
                console.log('[DEBUG] No youtubePlayers array found');
            }}
        }};
    }}
    // Debug panel to show active players
    if(!window.debugPanelCreated) {{
        window.debugPanelCreated = true;
        setInterval(function() {{
            var debugInfo = document.getElementById('debug-players-info');
            if(!debugInfo) {{
                debugInfo = document.createElement('div');
                debugInfo.id = 'debug-players-info';
                debugInfo.style.cssText = 'position:fixed;top:10px;right:10px;background:rgba(0,0,0,0.8);color:white;padding:10px;border-radius:5px;z-index:9999;font-size:12px;max-width:300px;';
                document.body.appendChild(debugInfo);
            }}
            var activePlayers = window.youtubePlayers ? window.youtubePlayers.length : 0;
            debugInfo.innerHTML = '<strong>🎥 Active Players:</strong> ' + activePlayers + '<br><small>Check console for detailed logs</small>';
        }}, 1000);
    }}
    // Video display fix - reload iframe if video appears black
    setTimeout(function() {{
        var iframe = document.getElementById('ytplayer-{vid_us}');
        if(iframe && player_{vid_us}) {{
            // Check if video is loaded but not displaying properly
            setTimeout(function() {{
                try {{
                    var state = player_{vid_us}.getPlayerState();
                    if(state === 1 || state === 2) {{ // Playing or Paused
                        console.log('[DEBUG] Video {vid_us} appears to be playing but may have display issues');
                        // Try to refresh the iframe
                        var src = iframe.src;
                        iframe.src = '';
                        setTimeout(function() {{
                            iframe.src = src;
                            console.log('[DEBUG] Reloaded iframe for {vid_us}');
                        }}, 100);
                    }}
                }} catch(e) {{
                    console.log('[DEBUG] Error checking player state for {vid_us}:', e);
                }}
            }}, 2000);
        }}
    }}, 1000);
    // If the API is already loaded, manually trigger the ready function for this new player instance.
    if(window.YT && window.YT.Player){{ onYouTubeIframeAPIReady(); }}
    """
    return _create_base_player_html(video_id=youtube_id, start_time=start_time, fps=fps, video_name=video_name, player_html=player_html, player_js_init=player_js_init, player_js_handlers=player_js_handlers)


def create_local_player_html(file_url: str, start_time: float, fps: float, video_name: str) -> str:
    vid_us = re.sub(r'[^a-zA-Z0-9_]', '_', video_name)
    player_html = f'<video id="fileplayer-{vid_us}" preload="metadata" playsinline controls crossorigin="anonymous"><source src="{file_url}" type="video/mp4" /></video>'
    player_js_init = ''
    player_js_handlers = f"""
    var player_{vid_us};
    document.addEventListener('DOMContentLoaded',function(){{ player_{vid_us}=document.getElementById('fileplayer-{vid_us}'); if(player_{vid_us}){{ player_{vid_us}.addEventListener('loadedmetadata',onPlayerReady_{vid_us}); player_{vid_us}.addEventListener('play',()=>onPlayerStateChange_{vid_us}(1)); player_{vid_us}.addEventListener('pause',()=>onPlayerStateChange_{vid_us}(2)); player_{vid_us}.addEventListener('ended',()=>onPlayerStateChange_{vid_us}(0)); }} }});
    function onPlayerReady_{vid_us}(){{
        document.getElementById('status-{vid_us}').textContent='Player Ready';
        player_{vid_us}.currentTime={start_time};
        try{{var d=player_{vid_us}.duration; if(d){{var mF=Math.floor(d*{fps});document.getElementById('seek-frame-{vid_us}').max=mF;document.getElementById('rs-range-min-{vid_us}').max=mF;document.getElementById('rs-range-max-{vid_us}').max=mF;document.getElementById('frame-right-{vid_us}').innerHTML='<strong>'+mF+' F</strong>';}} }}catch(_){{}}
        if(timeUpdateInterval_{vid_us})clearInterval(timeUpdateInterval_{vid_us}); timeUpdateInterval_{vid_us}=setInterval(updateTimeAndFrame_{vid_us},200);
    }}
    function onPlayerStateChange_{vid_us}(s){{ var n={{0:'Ended',1:'Playing',2:'Paused'}}; document.getElementById('player-state-{vid_us}').textContent=n[s]||'Buffering'; updateTimeAndFrame_{vid_us}();}}
    function updateTimeAndFrame_{vid_us}(){{
        if(!player_{vid_us})return;
        var t=player_{vid_us}.currentTime, f=calculateFrame(t);
        document.getElementById('current-time-{vid_us}').textContent=formatTimeMMSSFF(t,{fps});
        document.getElementById('current-frame-{vid_us}').textContent=f;
        var s=document.getElementById('seek-frame-{vid_us}'); if(s&&!window._seeking_{vid_us}){{s.value=f;document.getElementById('frame-left-{vid_us}').innerHTML='<strong>'+f+' F</strong>';}}
    }}
    """
    return _create_base_player_html(video_id=video_name, start_time=start_time, fps=fps, video_name=video_name, player_html=player_html, player_js_init=player_js_init, player_js_handlers=player_js_handlers)

def fallback_translate_and_create_aspects(query_vi: str) -> Tuple[List[Dict], Dict]:
    """Fallback: translate query to English and create aspects from each line"""
    try:
        # Split query into lines and translate each to English
        lines = [line.strip() for line in query_vi.split('\n') if line.strip()]
        translated_lines = []
        
        for line in lines:
            try:
                # Detect language and translate to English if needed
                translator = GoogleTranslator(source='auto', target='en')
                translated = translator.translate(line)
                translated_lines.append(translated)
            except Exception as e:
                print(f"Translation error for '{line}': {e}")
                translated_lines.append(line)  # Use original if translation fails
        
        # Create aspects from translated lines
        aspects = []
        for i, line in enumerate(translated_lines):
            aspect = {
                "name": f"aspect_{i+1}",
                "prompts": [line],  # Use translated line as single prompt
                "weight": 1.0
            }
            aspects.append(aspect)
        
        # Create minimal seeds (empty for fallback)
        seeds = {
            "unigrams": [],
            "phrases": []
        }
        
        return aspects, seeds
        
    except Exception as e:
        print(f"Fallback translation failed: {e}")
        # Ultimate fallback: single aspect with original query
        aspects = [{
            "name": "fallback_aspect",
            "prompts": [query_vi],
            "weight": 1.0
        }]
        seeds = {"unigrams": [], "phrases": []}
        return aspects, seeds

def get_fps_from_csv(video_id: str) -> float:
    """Get FPS from map-keyframes CSV file"""
    csv_paths = [
        f"{DATA_ROOT}b1/map-keyframes/{video_id}.csv",
        f"{DATA_ROOT}b2/map-keyframes/{video_id}.csv",
    ]

    for csv_path in csv_paths:
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'fps' in df.columns and len(df) > 0:
                    # Return the first FPS value (should be consistent)
                    fps = df['fps'].iloc[0]
                    return float(fps)
        except Exception as e:
            print(f"Error reading FPS from {csv_path}: {e}")
            continue

    # Default FPS if not found
    return 25.0

def get_watch_url(video_id: str) -> Optional[str]:
    # Try media-info directories for {video_id}.json with a 'watch_url' field
    media_info_dirs = [
        f"{DATA_ROOT}b1/media-info/",
        f"{DATA_ROOT}b2/media-info/",
    ]
    candidates = [Path(media_dir) / f"{video_id}.json" for media_dir in media_info_dirs]

    for p in candidates:
        try:
            if p.exists():
                data = json.loads(Path(p).read_text(encoding='utf-8'))
                url = data.get("watch_url")
                if url:
                    return url
        except Exception:
            continue
    return None

def get_video_info(video_id: str) -> Optional[dict]:
    """Get video information from media-info JSON files"""
    media_info_dirs = [
        f"{DATA_ROOT}b1/media-info/",
        f"{DATA_ROOT}b2/media-info/",
    ]
    candidates = [Path(media_dir) / f"{video_id}.json" for media_dir in media_info_dirs]

    for p in candidates:
        try:
            if p.exists():
                data = json.loads(Path(p).read_text(encoding='utf-8'))
                return data
        except Exception:
            continue
    return None

# Start the local video server in the background
if not st.session_state.get("_local_video_server_started"):
    LOCAL_VIDEO_ROOT.mkdir(exist_ok=True)
    flask_app = make_video_app(str(LOCAL_VIDEO_ROOT.resolve()))
    ensure_server_once(flask_app, port=8801)

# =============================== STREAMLIT UI ============================================
st.set_page_config(page_title="HCMC AI 2025 – CLIP + BM25 Search", layout="wide")

# Row 1: Title
st.title("HCMC AI 2025 – CLIP + BM25 Search")
st.markdown("*Choose your search mode: **CLIP visual search** or **BM25 text search***")

# Row 2: Filter options (collapsible)
with st.expander("🔧 Search Filters", expanded=False):
    # Dataset selection
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Dataset Selection:**")
        batch_label = st.selectbox("Choose Dataset:", ["Both Batches", "Batch 1", "Batch 2"], index=0)

    with col2:
        st.markdown("**Advanced Filters:**")

    # Filter inputs
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        exclude_series = st.text_input(
            "❌ Exclude Series:",
            placeholder="L01,L02,K01",
            help="Exclude video series (comma-separated). E.g., L01,L02 to exclude batch 1 videos",
            key="exclude_series"
        )

    with filter_col2:
        exclude_videos = st.text_input(
            "❌ Exclude Videos:",
            placeholder="L01V001,L01V002",
            help="Exclude specific videos by ID (comma-separated)",
            key="exclude_videos"
        )

    # Parse filter inputs
    exclude_series_list = [s.strip() for s in exclude_series.split(',') if s.strip()]
    exclude_videos_list = [v.strip() for v in exclude_videos.split(',') if v.strip()]

    # Apply dataset selection
if batch_label == "Batch 1":
    DATASETS = {"b1"}
elif batch_label == "Batch 2":
    DATASETS = {"b2"}
else:
    DATASETS = {"b1", "b2"}

    # Show active filters summary
    active_filters = []
    if batch_label != "Both Batches":
        active_filters.append(f"Dataset: {batch_label}")
    if exclude_series_list:
        active_filters.append(f"Exclude {len(exclude_series_list)} series")
    if exclude_videos_list:
        active_filters.append(f"Exclude {len(exclude_videos_list)} videos")

    if active_filters:
        st.success(f"✅ **Active Filters:** {', '.join(active_filters)}")
    else:
        st.info("ℹ️ **No filters applied** - searching all available videos")

# Row 2.5: Gemini API Configuration and Testing
with st.expander("🔑 Gemini API Configuration", expanded=False):
    st.markdown("**Configure and test your Gemini API connection**")
    
    col1, col2 = st.columns([3, 1])
    
    # API Key input
    api_key = st.text_input(
        "Gemini API Key:",
        value=GOOGLE_API_KEY,
        type="password",
        help="Enter your Google Gemini API key. You can get one from https://aistudio.google.com"
    )

    # Update the global API key if changed
    if api_key and api_key != GOOGLE_API_KEY:
        GOOGLE_API_KEY = api_key
        os.environ["GOOGLE_API_KEY"] = api_key

    # Gemini Configuration Options (only show if API key is configured)
    if api_key and api_key.strip():
        st.markdown("**Model Selection:**")
        col_model = st.columns([1])[0]

        with col_model:
            gemini_model = st.selectbox(
                "Gemini Model:",
                ["gemini-2.5-flash", "gemini-2.5-pro"],
                index=0,
                help="gemini-2.5-pro: High quality, gemini-2.5-flash: Fast speed but sometime get bad result"
            )

        st.markdown("**Generation Parameters:**")
        col_temp, col_topp = st.columns(2)

        with col_temp:
            gemini_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
                help="Controls randomness. Lower = more focused, Higher = more creative"
            )

        with col_topp:
            gemini_top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.05,
                help="Controls diversity. Lower = more focused, Higher = more diverse"
            )

    test_button = st.button("🧪 Test Gemini API", type="primary")
    
    # Test API connection (using google-genai Client)
    if test_button:
        if not api_key:
            st.error("❌ Please enter your Gemini API key first!")
        else:
            with st.spinner("Testing Gemini API connection..."):
                try:
                    # Test with a simple prompt
                    test_prompt = "Hello! Please respond with 'API connection successful' if you can read this message."

                    # Use the new google-genai SDK consistently
                    from google import genai
                    client = genai.Client(api_key=api_key)
                    resp = client.models.generate_content(
                        model='gemini-2.0-flash',
                        contents=test_prompt
                    )
                    response_text = (getattr(resp, "text", None) or getattr(resp, "output_text", None) or "").strip()

                    if response_text:
                        st.success("✅ **Gemini API Test Successful!**")
                        st.info(f"**Response:** {response_text}")
                        st.markdown("🎉 Your Gemini API is working correctly and ready to use!")
                    else:
                        st.warning("⚠️ **API responded but with empty content**")
                        st.markdown("The API connection works but returned no content.")

                except Exception as e:
                    st.error("❌ **Gemini API Test Failed!**")
                    st.error(f"**Error:** {str(e)}")
                    st.markdown("**Common issues:**")
                    st.markdown("- Invalid API key")
                    st.markdown("- API quota exceeded")
                    st.markdown("- Network connectivity issues")
                    st.markdown("- API service temporarily unavailable")
    
    # Show current status
    if GOOGLE_API_KEY:
        st.success("🔑 **API Key Status:** Configured")
    else:
        st.warning("⚠️ **API Key Status:** Not configured")

# Row 3: Setting
with st.expander("⚙️ Setting", expanded=False):
    st.markdown("**Time Settings**")
    # Apply pending inputs BEFORE widgets are created
    _pending = st.session_state.pop('pending_setting_inputs', None)
    if _pending:
        # Clear previous widget states so value= is used
        for _k in ['win_str_inp', 'skip_str_inp', 'del_str_inp', 'topk_inp']:
            try:
                st.session_state.pop(_k)
            except Exception:
                pass
    _win_default = (_pending.get('win') if _pending else sec_to_mmss(float(st.session_state.get('WINDOW_SEC', WINDOW_SEC))))
    _skip_default = (_pending.get('skip') if _pending else sec_to_mmss(float(st.session_state.get('SKIP_SEC', SKIP_SEC))))
    _del_default = (_pending.get('del') if _pending else sec_to_mmss(float(st.session_state.get('DELTA_SEC', DELTA_SEC))))
    _top_default = (_pending.get('top') if _pending else int(st.session_state.get('TOP_VIDEOS', TOP_VIDEOS)))

    col_win, col_skip, col_del, col_top = st.columns([1, 1, 1, 1])
    with col_win:
        win_str_inp = st.text_input(
            "Window Second (mm:ss)",
            value=st.session_state.get('win_str_inp', _win_default),
            key="win_str_inp",
            help="Độ dài cửa sổ trượt khi tính CLIP window score"
        )
    with col_skip:
        skip_str_inp = st.text_input(
            "Skip Intro (mm:ss)",
            value=st.session_state.get('skip_str_inp', _skip_default),
            key="skip_str_inp",
            help="Skip phần intro của video"
        )
    with col_del:
        del_str_inp = st.text_input(
            "Rerank Window (mm:ss)",
            value=st.session_state.get('del_str_inp', _del_default),
            key="del_str_inp",
            help="Mở rộng cửa sổ t0/t1 khi rerank transcript"
        )
    with col_top:
        topk_inp = st.number_input(
            "Top K",
            min_value=1,
            max_value=200,
            value=int(st.session_state.get('topk_inp', _top_default)),
            step=1,
            key="topk_inp",
            help="Số video hiển thị tối đa"
        )
    # Show Update button only when any value differs
    pending_change = (
        sec_to_mmss(float(st.session_state.get('WINDOW_SEC', WINDOW_SEC))) != win_str_inp or
        sec_to_mmss(float(st.session_state.get('SKIP_SEC', SKIP_SEC))) != skip_str_inp or
        sec_to_mmss(float(st.session_state.get('DELTA_SEC', DELTA_SEC))) != del_str_inp or
        int(st.session_state.get('TOP_VIDEOS', TOP_VIDEOS)) != int(topk_inp)
    )
    col_sp, col_btn_upd, col_btn_reset, col_btn_def = st.columns([5, 1, 1, 1])
    with col_btn_upd:
        update_cfg = st.button("Update Setting", type="secondary") if pending_change else None
    with col_btn_reset:
        reset_cfg = st.button("Reset Setting", type="secondary") if pending_change else None
    with col_btn_def:
        default_cfg = st.button("Default Setting", type="secondary")

    if update_cfg:
        st.session_state['WINDOW_SEC'] = parse_mmss(win_str_inp, WINDOW_SEC)
        st.session_state['SKIP_SEC'] = parse_mmss(skip_str_inp, SKIP_SEC)
        st.session_state['DELTA_SEC'] = parse_mmss(del_str_inp, DELTA_SEC)
        st.session_state['TOP_VIDEOS'] = int(topk_inp)
        st.rerun()
    if 'reset_cfg' in locals() and reset_cfg:
        # Set pending inputs to current applied settings and rerun
        _w = float(st.session_state.get('WINDOW_SEC', WINDOW_SEC))
        _s = float(st.session_state.get('SKIP_SEC', SKIP_SEC))
        _d = float(st.session_state.get('DELTA_SEC', DELTA_SEC))
        _t = int(st.session_state.get('TOP_VIDEOS', TOP_VIDEOS))
        st.session_state['pending_setting_inputs'] = {
            'win': sec_to_mmss(_w),
            'skip': sec_to_mmss(_s),
            'del': sec_to_mmss(_d),
            'top': _t,
        }
        st.rerun()
    if 'default_cfg' in locals() and default_cfg:
        # Restore defaults
        st.session_state['WINDOW_SEC'] = 40.0
        st.session_state['SKIP_SEC'] = 0.0
        st.session_state['DELTA_SEC'] = 20.0
        st.session_state['TOP_VIDEOS'] = 20
        st.session_state['pending_setting_inputs'] = {
            'win': sec_to_mmss(40.0),
            'skip': sec_to_mmss(0.0),
            'del': sec_to_mmss(20.0),
            'top': 20,
        }
        st.rerun()

# Row 4: Query input
col_max_aspects, col_mode = st.columns([1.2, 2])

# Apply runtime overrides from session so settings affect ranking
try:
    WINDOW_SEC = float(st.session_state.get('WINDOW_SEC', WINDOW_SEC))
    SKIP_SEC = float(st.session_state.get('SKIP_SEC', SKIP_SEC))
    DELTA_SEC = float(st.session_state.get('DELTA_SEC', DELTA_SEC))
    TOP_VIDEOS = int(st.session_state.get('TOP_VIDEOS', TOP_VIDEOS))
except Exception:
    pass

with col_max_aspects:
    max_aspects_input = st.text_input(
        "🎯 Max aspects:",
        value="",
        placeholder="Không nhập thì ngẫu nhiên",
        help="Số lượng aspects Gemini tạo (để trống = tự động)",
        key="max_aspects_input"
    )

with col_mode:
    search_mode = st.selectbox(
        "🔎 Search Mode",
        [
            "CLIP Multi-aspects Search",
            "CLIP Multi-aspects Search + Rerank",
            "Transcript Seed Search",
            "BM25 Keyword Search",
            "Search by Video ID",
        ],
        index=0,
    )

# Derive mode flags from selection
rerank_enabled = (search_mode == "CLIP Multi-aspects Search + Rerank")
transcript_seed_enabled = (search_mode == "Transcript Seed Search")
bm25_enabled = (search_mode == "BM25 Keyword Search")
video_id_enabled = (search_mode == "Search by Video ID")

# Reset run/caches when switching modes to prevent unintended auto-runs
if 'last_mode' not in st.session_state:
    st.session_state['last_mode'] = search_mode
elif st.session_state['last_mode'] != search_mode:
    st.session_state['run_flag'] = False
    st.session_state['auto_run_after_enhance'] = False
    st.session_state['last_mode'] = search_mode

# Gemini availability toggle (placed below Max aspects and above query input)
gemini_configured = bool(GOOGLE_API_KEY and GOOGLE_API_KEY.strip())
turn_off_gemini = st.checkbox(
    "🔌 Turn off Gemini",
    value=False,
    key="turn_off_gemini",
    help="Disable Gemini API to skip Enhance step. Rerank requires Gemini and will be disabled."
)
gemini_available = gemini_configured and not turn_off_gemini

# If Gemini is not available, prevent selecting rerank mode
if (not gemini_available) and rerank_enabled:
    st.warning("⚠️ Rerank requires Gemini. Switching to basic CLIP search.")
    search_mode = "CLIP Multi-aspects Search"
    rerank_enabled = False
    bm25_enabled = (search_mode == "BM25 Keyword Search")

# BM25 options when enabled
if bm25_enabled:
    col_help = st.columns([1])[0]

    with col_help:
        with st.popover("📚 BM25 Query Syntax"):
            st.markdown("""
            ### 🔍 **BM25 Query Syntax**

            #### **SQL Operators:**
            ```sql
            -- Basic Operations
            word1 AND word2    -- Both words required
            word1 OR word2     -- At least one word
            word1 NOT word2    -- First word without second

            -- Shortcuts (same as above)
            word1 & word2      -- Same as AND
            word1 | word2      -- Same as OR

            -- Exact Phrase Matching
            "exact phrase"     -- Match exact sequence
            ```

            #### **📝 Examples:**
            ```sql
            -- Simple queries
            cát AND điêu              -- Videos with both words
            cát OR biển               -- Videos with either word
            cát NOT biển              -- Videos with "cát" but not "biển"

            -- Phrase queries
            "điêu khắc cát"           -- Exact phrase match
            "công an"                 -- Match "công an" as complete phrase

            -- Complex queries
            "cột đá" AND (10 OR 12)   -- Phrase + numbers
            cát AND "điêu khắc" NOT biển  -- Mixed operators
            ```

            #### **⚡ Quick Reference:**
            | Operator | SQL | Example | Description |
            |----------|-----|---------|-------------|
            | `AND` | `AND` | `cát AND điêu` | Both words required |
            | `OR` | `OR` | `cát OR biển` | At least one word |
            | `NOT` | `NOT` | `cát NOT biển` | First without second |
            | `&` | `AND` | `cát & điêu` | Same as AND |
            | `\\|` | `OR` | `cát \\| biển` | Same as OR |
            | `"phrase"` | `"phrase"` | `"công an"` | Exact phrase match |
            """)


# Video ID input when enabled
if video_id_enabled:
    st.markdown("**Video ID Search:**")
    video_id_input = st.text_input(
        "Enter Video ID:",
        placeholder="e.g., K01_V001, L02_V005",
        help="Enter the video ID to search for (e.g., K01_V001, L02_V005)",
        key="video_id_input"
    )
else:
    video_id_input = ""

# Dynamic placeholder based on search mode
if bm25_enabled:
    placeholder_text = """Ví dụ BM25 queries:
• cát AND điêu AND khắc (tìm video có cả 3 từ)
• cát OR "điêu khắc" (tìm video có "cát" hoặc cụm "điêu khắc")
• cát NOT biển (tìm video có "cát" nhưng không có "biển")

Nhấn nút "📚 BM25 Query Syntax" để xem chi tiết syntax."""
else:
    placeholder_text = "Nhập miêu tả truy vấn tiếng Việt / English ..."

# Only show query input if not in video ID mode
if not video_id_enabled:
    query_vi = st.text_area("Enter Query:", height=150, placeholder=placeholder_text)
else:
    query_vi = ""  # Set empty query for video ID mode

# Button styling for full width
st.markdown(
    """
    <style>
    .stButton > button {
        width: 100% !important;
        font-size: 18px !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        margin: 0 !important;
        padding: 0.75rem 1rem !important;
        display: block !important;
    }
    .stButton {
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* BM25 Highlight Styles */
    .bm25-highlight {
        background-color: #fff3cd !important;
        color: #856404 !important;
        font-weight: bold !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
        border: 1px solid #ffeaa7 !important;
    }

    .bm25-fulltext {
        line-height: 1.6 !important;
        margin: 8px 0 !important;
        max-height: 200px !important;
        overflow-y: auto !important;
        padding: 10px !important;
        background-color: #f8f9fa !important;
        color: #212529 !important;
        border: 1px solid #e9ecef !important;
        border-radius: 5px !important;
        font-size: 14px !important;
    }

    .bm25-fulltext b, .bm25-fulltext strong {
        background-color: #fff3cd !important;
        color: #212529 !important;
        font-weight: bold !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
        border: 1px solid #ffeaa7 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for query enhancement
if 'query_enhanced' not in st.session_state:
    st.session_state.query_enhanced = False
if 'enhanced_data' not in st.session_state:
    st.session_state.enhanced_data = None

# Row 5: Query buttons - Conditional display based on search mode
# Persist run flag across reruns so UI remains in results state
if 'run_flag' not in st.session_state:
    st.session_state['run_flag'] = False
run = st.session_state['run_flag']

# Check if Gemini API is configured
gemini_configured = bool(GOOGLE_API_KEY and GOOGLE_API_KEY.strip())
gemini_available = 'turn_off_gemini' in st.session_state and gemini_configured and (not st.session_state['turn_off_gemini'])

# Auto-run once after first enhance (so buttons switch to Run/Reset immediately)
if st.session_state.get('auto_run_after_enhance', False):
    run = True
    st.session_state['run_flag'] = True  # persist results state across reruns
    st.session_state['auto_run_after_enhance'] = False

# Default values for Gemini parameters (when API is not configured)
gemini_temperature = 0.7
gemini_top_p = 0.9

if bm25_enabled:
    # BM25 mode - always show Run Query directly (no enhancement needed)
    run_button = st.button("🚀 Run Query", width='stretch', type="primary")
    if run_button:
        # Reset watch mode keys when starting new search
        keys_to_reset = [key for key in st.session_state.keys() if key.startswith("watch_mode_")]
        for key in keys_to_reset:
            del st.session_state[key]
            
        if not query_vi.strip():
            st.error("Please enter a query first!")
        else:
            run = True; st.session_state['run_flag'] = True
            st.info("🔍 **BM25 Keyword Search** - Searching directly by keywords")
elif transcript_seed_enabled:
    # Transcript Seed Search mode - always show Run Query directly (no enhancement needed)
    run_button = st.button("🔍 Run Transcript Seed Search", width='stretch', type="primary")
    if run_button:
        # Reset watch mode keys when starting new search
        keys_to_reset = [key for key in st.session_state.keys() if key.startswith("watch_mode_")]
        for key in keys_to_reset:
            del st.session_state[key]
            
        if not query_vi.strip():
            st.error("Please enter a query first!")
        else:
            run = True; st.session_state['run_flag'] = True
            st.info("🔍 **Transcript Seed Search** - Searching using Gemini-generated seeds only")
elif video_id_enabled:
    # Video ID mode - always show Search Video directly (no enhancement needed)
    run_button = st.button("🔍 Search Video", width='stretch', type="primary")
    if run_button:
        # Reset watch mode keys when starting new search
        keys_to_reset = [key for key in st.session_state.keys() if key.startswith("watch_mode_")]
        for key in keys_to_reset:
            del st.session_state[key]
            
        if not video_id_input.strip():
            st.error("Please enter a video ID first!")
        else:
            run = True; st.session_state['run_flag'] = True
            st.info("🔍 **Video ID Search** - Searching for specific video")
elif gemini_available:
    # CLIP mode with Gemini - show Enhance Query first, then Run Query
    if not st.session_state.query_enhanced:
        # Show Enhance Query button (Green, full width)
        enhance_button = st.button("🌱 Enhance Query", width='stretch', type="primary")
        if enhance_button:
            # Reset watch mode keys when starting new search
            keys_to_reset = [key for key in st.session_state.keys() if key.startswith("watch_mode_")]
            for key in keys_to_reset:
                del st.session_state[key]
                
            if not query_vi.strip():
                st.error("Please enter a query first!")
            else:
                with st.spinner("Enhancing query with Gemini..."):
                    try:
                        enhanced_data = extract_aspects_and_seeds_with_gemini(
                            query_vi, gemini_temperature, gemini_top_p,
                            gemini_model if 'gemini_model' in locals() else "gemini-2.5-flash",
                            max_aspects_input.strip() if max_aspects_input.strip() else "N"
                        )
                        st.session_state.enhanced_data = enhanced_data
                        st.session_state.query_enhanced = True
                        st.success("✅ Query enhanced successfully! Running search...")
                        # Trigger a rerun where Run/Reset buttons show and auto-run executes once
                        st.session_state['auto_run_after_enhance'] = True
                        st.session_state['run_flag'] = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to enhance query: {e}")
    else:
        # After enhancement: show Run and Reset so user can reuse aspects/seeds for rerank later
        cols_rr = st.columns([1, 1])
        with cols_rr[0]:
            run_button = st.button("🚀 Run Query", width='stretch', type="secondary")
        with cols_rr[1]:
            reset_button = st.button("🔄 Reset Query", width='stretch')

        if run_button:
            # Reset watch mode keys when starting new search
            keys_to_reset = [key for key in st.session_state.keys() if key.startswith("watch_mode_")]
            for key in keys_to_reset:
                del st.session_state[key]
                
            run = True; st.session_state['run_flag'] = True
        if reset_button:
            st.session_state.query_enhanced = False
            st.session_state.enhanced_data = None
            st.session_state['run_flag'] = False
            st.session_state.pop('clip_cache', None)
            st.rerun()
else:
    # CLIP mode without Gemini - show Run Query directly with fallback
    run_button = st.button("🚀 Run Query", width='stretch', type="primary")
    if run_button:
        # Reset watch mode keys when starting new search
        keys_to_reset = [key for key in st.session_state.keys() if key.startswith("watch_mode_")]
        for key in keys_to_reset:
            del st.session_state[key]
            
        if not query_vi.strip():
            st.error("Please enter a query first!")
        else:
            run = True; st.session_state['run_flag'] = True
            # Use fallback translation directly
            st.info("ℹ️ **Using Fallback Translation** - Configure Gemini API for better results")

# Row 5: Logs expander
log_exp = st.expander("📝 Logs", expanded=False)
log_container = log_exp.empty()

def run_with_logs(fn, *args, **kwargs):
    buf = io.StringIO()
    try:
        import contextlib
        with contextlib.redirect_stdout(buf):
            return fn(*args, **kwargs), buf
    except Exception as e:
        print(f"[ERROR] {fn.__name__}: {e}")
        return None, buf
    finally:
        pass

# Row 8 & 9: Results areas (placeholders)
clip_results_container = st.empty()
rerank_results_container = st.empty()

# (Dynamic sidebar for sorting will be rendered after results are available in CLIP mode only)

if run:
    # Check if BM25 search is enabled
    if bm25_enabled:
        # BM25 Keyword Search Mode
        with log_container.container():
            st.text("🔍 Running BM25 keyword search...")

        try:
            with st.spinner("🔎 Đang search BM25..."):
                bm25_results = bm25_search_videos(query_vi, limit=TOP_VIDEOS)

                # Apply search filters
                bm25_results = apply_search_filters(bm25_results, exclude_series_list, exclude_videos_list)

                # Remove duplicate results that are too close in time
                bm25_results = deduplicate_bm25_results(bm25_results, time_threshold=30.0)

            # Highlight is always enabled for BM25 results

            with log_container.container():
                st.text("🔍 Running BM25 keyword search...")
                st.text(f"Found {len(bm25_results)} results")

            if bm25_results:
                # Display BM25 results
                with clip_results_container.container():
                    st.subheader("🔍 BM25 Keyword Search Results")

                    for i, result in enumerate(bm25_results):
                        vid = result["video_id"]
                        score = result["score"]
                        snippet = result["snippet"]
                        timestamp = result["timestamp"]

                        with st.container():
                            # Header with keyframe
                            col_img, col_info = st.columns([1, 3])

                            with col_img:
                                # Find keyframe closest to timestamp
                                target_keyframe = timestamp_to_keyframe_number(vid, timestamp)
                                keyframe_img, actual_frame = find_best_keyframe(vid, target_keyframe)

                                if keyframe_img is not None and keyframe_img.size[0] > 100:
                                    st.image(keyframe_img, caption=f"Keyframe {actual_frame}", width='stretch')
                                else:
                                    # Fallback image if keyframe not found
                                    fallback_img = Image.fromarray(np.full((180, 320, 3), 200, dtype=np.uint8))
                                    st.image(fallback_img, caption=f"No keyframe at ~{sec_to_mmss(timestamp)}", width='stretch')

                            with col_info:
                                st.markdown(f"**[{i+1}] {vid}** · BM25 Score: {score:.3f}")
                                st.markdown(f"**Timestamp:** {sec_to_mmss(timestamp)} | **Keyframe:** {actual_frame}")
                                st.markdown(f"**Source:** {result['db_source'].upper()}")

                                # Display highlighted full text
                                highlighted_text = result.get("highlighted_full_text", snippet)
                                st.markdown(f"""
                                    <div class="bm25-fulltext">
                                        <strong>Full Text:</strong> {highlighted_text}
                                    </div>
                                """, unsafe_allow_html=True)

                        # Try to get watch URL and display video player
                        url = get_watch_url(vid)
                        with st.popover("🎥 Watch Video", width='stretch'):
                            st.markdown(create_popover_css(), unsafe_allow_html=True)
                            col1, col2 = st.columns([2, 1]) # Căn lề radio qua phải
                            with col1:
                                st.markdown("")  # Cột trống để đẩy sang phải
                            with col2:
                                # Danh sách các lựa chọn (giữ logic đơn giản)
                                options = ("Youtube Online", "Kaggle Offline")
                                
                                # Hàm để định dạng cách hiển thị (thêm icon)
                                def format_func(option):
                                    if option == "Youtube Online":
                                        return f"▶️ {option}"
                                    elif option == "Kaggle Offline":
                                        return f"☁️ {option}"
                                    return option

                                watch_mode = st.radio(
                                    "Watch Mode Select",
                                    options,
                                    key=f"watch_mode_bm25_{vid}_{i}",
                                    horizontal=True,
                                    label_visibility="collapsed",
                                    format_func=format_func
                                )

                            start_sec = int(max(0, timestamp))
                            actual_video_id = vid
                            actual_fps = get_fps_from_csv(actual_video_id)
                            
                            if watch_mode == "Youtube Online":
                                url = get_watch_url(vid)
                                if url:
                                    youtube_video_id = url.split('v=')[1].split('&')[0] if 'v=' in url else url.split('/')[-1]
                                    components.html(create_youtube_player_html(youtube_video_id, start_sec, fps=actual_fps, video_name=actual_video_id), height=1500)
                                else:
                                    st.warning("YouTube URL not found for this video. Try Kaggle Offline mode.")
                            
                            elif watch_mode == "Kaggle Offline":
                                # Chỉ cần một placeholder duy nhất cho spinner và các thông báo
                                status_placeholder = st.empty()
                                
                                local_path = download_video_from_kaggle(actual_video_id, LOCAL_VIDEO_ROOT, status_placeholder)
                                
                                if local_path:
                                    file_url = f"{LOCAL_VIDEO_URL_BASE}/{local_path.name}"
                                    components.html(create_local_player_html(file_url, start_sec, fps=actual_fps, video_name=actual_video_id), height=1500, scrolling=True)
                                else:
                                    # Error message is already shown by the download function
                                    pass

                        st.divider()
            else:
                st.warning("No BM25 search results found. Try different keywords or check if BM25 databases exist.")

        except Exception as e:
            st.error(f"BM25 search failed: {e}")
            st.info("Make sure BM25 databases exist in the data directory")

        # Don't continue with CLIP search when BM25 is enabled
        st.stop()

    # Transcript Seed Search Mode
    elif transcript_seed_enabled:
        with log_container.container():
            st.text("🔍 Running Transcript Seed Search...")

        try:
            with st.spinner("🔎 Đang search transcript seeds..."):
                # Get Gemini aspects and seeds
                if gemini_available:
                    with st.spinner("🤖 Enhancing query with Gemini..."):
                        try:
                            enhanced_data = extract_aspects_and_seeds_with_gemini(
                                query_vi, gemini_temperature, gemini_top_p,
                                gemini_model if 'gemini_model' in locals() else "gemini-2.5-flash",
                                max_aspects_input.strip() if max_aspects_input.strip() else "N"
                            )
                            spec = enhanced_data
                            gemini_log = f"Query enhanced successfully with Gemini"
                        except Exception as e:
                            err = str(e)
                            spec = None
                            gemini_log = f"Query enhance failed — using fallback translation{' | error: ' + err if err else ''}"
                else:
                    # Use fallback translation
                    spec = None
                    gemini_log = "Using fallback translation method (Gemini off or not configured)"

                if spec:
                    ASPECTS = spec.get("aspects", [])
                    SEEDS = spec.get("seeds", {})
                else:
                    # Fallback: create minimal aspects and seeds
                    aspects, seeds = fallback_translate_and_create_aspects(query_vi)
                    ASPECTS = aspects
                    SEEDS = seeds

                # Run transcript seed search
                transcript_results = transcript_seed_search(
                    query_vi, SEEDS,
                    top_candidates=100,
                    delta_sec=10.0,
                    text_weights={"any": 0.3, "phr": 0.7},
                    verbose=True, timing=True,
                    workers=8, phrases_only=False, max_unigrams=16, max_phrases=10
                )

            with log_container.container():
                st.text("🔍 Running Transcript Seed Search...")
                st.text(f"Found {len(transcript_results)} results")
                st.text(f"Seeds used: {SEEDS}")

            if not transcript_results.empty:
                # Display transcript seed results
                with clip_results_container.container():
                    st.subheader("🔍 Transcript Seed Search Results")

                    for i, row in transcript_results.head(TOP_VIDEOS).iterrows():
                        vid = row["video_id"]
                        hits_any = row["hits_any"]
                        hits_phr = row["hits_phr"]
                        text_score = row["text_score"]
                        fused = row["fused"]

                        st.markdown(f"**[{i+1}] {vid}** · Text Score: {fused:.3f} (Any: {hits_any}, Phr: {hits_phr})")

                        # Get video info
                        video_info = get_video_info(vid)
                        if video_info:
                            if 'title' in video_info:
                                st.markdown(f"**Title:** {video_info['title']}")
                            if 'description' in video_info:
                                st.markdown(f"**Description:** {video_info['description'][:200]}...")

                        # Watch Video button with popover
                        with st.popover("🎥 Watch Video", width='stretch'):
                            st.markdown(create_popover_css(), unsafe_allow_html=True)
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown("")
                            with col2:
                                options = ("Youtube Online", "Kaggle Offline")
                                def format_func(option):
                                    if option == "Youtube Online":
                                        return f"▶️ {option}"
                                    elif option == "Kaggle Offline":
                                        return f"☁️ {option}"
                                    return option

                                watch_mode = st.radio(
                                    "Watch Mode Select",
                                    options,
                                    key=f"watch_mode_transcript_{vid}",
                                    horizontal=True,
                                    label_visibility="collapsed",
                                    format_func=format_func
                                )

                            start_sec = 0
                            actual_video_id = vid
                            actual_fps = get_fps_from_csv(actual_video_id)
                            
                            if watch_mode == "Youtube Online":
                                url = get_watch_url(actual_video_id)
                                if url:
                                    youtube_video_id = url.split('v=')[1].split('&')[0] if 'v=' in url else url.split('/')[-1]
                                    components.html(create_youtube_player_html(youtube_video_id, start_sec, fps=actual_fps, video_name=actual_video_id), height=1500)
                                else:
                                    st.warning(f"YouTube URL not found for {actual_video_id}. Try Kaggle Offline mode.")

                            elif watch_mode == "Kaggle Offline":
                                status_placeholder = st.empty()
                                
                                local_path = download_video_from_kaggle(actual_video_id, LOCAL_VIDEO_ROOT, status_placeholder)
                                
                                if local_path:
                                    file_url = f"{LOCAL_VIDEO_URL_BASE}/{local_path.name}"
                                    components.html(create_local_player_html(file_url, start_sec, fps=actual_fps, video_name=actual_video_id), height=1500, scrolling=True)
                        
                        st.divider()
            else:
                st.warning("No transcript seed search results found. Try different query or check if transcript files exist.")

        except Exception as e:
            st.error(f"Transcript seed search failed: {e}")
            st.info("Make sure transcript files exist in the data directory")

        # Don't continue with CLIP search when transcript seed is enabled
        st.stop()

    # Video ID Search Mode
    elif video_id_enabled:
        if not video_id_input.strip():
            st.error("Please enter a video ID first!")
        else:
            with log_container.container():
                st.text(f"🔍 Searching for video ID: {video_id_input}")
            
            try:
                with st.spinner(f"🔎 Searching for video {video_id_input}..."):
                    video_info = get_video_info(video_id_input)
                    
                    if video_info:
                        st.success(f"✅ Found video: {video_id_input}")
                        st.markdown("### 📹 Video Information")
                        col_thumb, col_info = st.columns([0.3, 1])
                        
                        with col_thumb:
                            if 'thumbnail_url' in video_info and video_info['thumbnail_url']:
                                st.image(video_info['thumbnail_url'], caption="Video Thumbnail", width=400)
                            else:
                                st.markdown("*No thumbnail available*")
                        
                        with col_info:
                            if 'title' in video_info: st.markdown(f"**Title:** {video_info['title']}")
                            if 'author' in video_info: st.markdown(f"**Author:** {video_info['author']}")
                            if 'length' in video_info:
                                length_seconds = int(video_info['length'])
                                minutes, seconds = divmod(length_seconds, 60)
                                st.markdown(f"**Length:** {minutes:02d}:{seconds:02d}")
                            if 'publish_date' in video_info: st.markdown(f"**Publish Date:** {video_info['publish_date']}")
                        
                        # --- CORRECTED VIDEO PLAYER LOGIC FOR VIDEO ID SEARCH ---
                        st.markdown("### 🎥 Video Player")
                        with st.popover("🎥 Watch Video", width='stretch'):
                            st.markdown(create_popover_css(), unsafe_allow_html=True)
                            col1, col2 = st.columns([2, 1]) # Căn lề radio qua phải
                            with col1:
                                st.markdown("")  # Cột trống để đẩy sang phải
                            with col2:
                                options = ("Youtube Online", "Kaggle Offline")
                                def format_func(option):
                                    if option == "Youtube Online":
                                        return f"▶️ {option}"
                                    elif option == "Kaggle Offline":
                                        return f"☁️ {option}"
                                    return option

                                watch_mode = st.radio(
                                    "Watch Mode Select",
                                    options,
                                    key=f"watch_mode_video_id_{video_id_input}",
                                    horizontal=True,
                                    label_visibility="collapsed",
                                    format_func=format_func
                                )

                            start_sec = 0
                            actual_video_id = video_id_input
                            actual_fps = get_fps_from_csv(actual_video_id)
                            
                            if watch_mode == "Youtube Online":
                                url = get_watch_url(actual_video_id)
                                if url:
                                    youtube_video_id = url.split('v=')[1].split('&')[0] if 'v=' in url else url.split('/')[-1]
                                    components.html(create_youtube_player_html(youtube_video_id, start_sec, fps=actual_fps, video_name=actual_video_id), height=1500)
                                else:
                                    st.warning(f"YouTube URL not found for {actual_video_id}. Try Kaggle Offline mode.")

                            elif watch_mode == "Kaggle Offline":
                                status_placeholder = st.empty()
                                
                                local_path = download_video_from_kaggle(actual_video_id, LOCAL_VIDEO_ROOT, status_placeholder)
                                
                                if local_path:
                                    file_url = f"{LOCAL_VIDEO_URL_BASE}/{local_path.name}"
                                    components.html(create_local_player_html(file_url, start_sec, fps=actual_fps, video_name=actual_video_id), height=1500, scrolling=True)
                        
                        if 'description' in video_info:
                            st.markdown("### 📝 Description")
                            st.markdown(video_info['description'])
                        
                        if 'keywords' in video_info and video_info['keywords']:
                            st.markdown("### 🏷️ Keywords")
                            st.markdown(", ".join(video_info['keywords']))
                        
                        if 'channel_url' in video_info:
                            st.markdown(f"### 🔗 Channel URL: {video_info['channel_url']}")
                        
                    else:
                        st.error(f"❌ Video ID '{video_id_input}' not found in media-info files")
                        
            except Exception as e:
                st.error(f"Video ID search failed: {e}")
            
        st.stop()

    # Prefer cached enhanced data if available
    if st.session_state.enhanced_data:
        spec = st.session_state.enhanced_data
        if spec.get("source") == "gemini":
            gemini_log = "Query enhanced successfully with Gemini API (cached)"
        else:
            err = spec.get("error", "")
            gemini_log = f"Query enhance failed — using fallback translation (cached){' | error: ' + err if err else ''}"
    elif gemini_available:
        # No cached enhancement; call Gemini normally
        with log_container.container():
            st.text("Running extract_aspects_and_seeds_with_gemini() ...")
        (spec, buf) = run_with_logs(
            extract_aspects_and_seeds_with_gemini,
            query_vi, gemini_temperature, gemini_top_p,
            gemini_model if 'gemini_model' in locals() else "gemini-2.5-flash",
            max_aspects_input.strip() if max_aspects_input.strip() else "N"
        )
        if spec is None:
            st.error("Failed to extract aspects/seeds. See logs.")
            st.stop()
        if spec.get("source") == "gemini":
            gemini_log = "Query enhanced successfully with Gemini API"
        else:
            err = spec.get("error", "")
            gemini_log = f"Query enhance failed — using fallback translation{' | error: ' + err if err else ''}"
    else:
        # Gemini is turned off or not available: force fallback
        if rerank_enabled:
            st.warning("⚠️ Rerank not available when Gemini is turned off. Running basic CLIP search.")
            rerank_enabled = False
        with log_container.container():
            st.text("Using fallback translation method...")
        aspects, seeds = fallback_translate_and_create_aspects(query_vi)
        spec = {"aspects": aspects, "seeds": seeds}
        gemini_log = "Using fallback translation method (Gemini off or not configured)"

    # Show Gemini call result first
    with log_container.container():
        st.text(gemini_log)
    
    # Then show the extracted aspects
    ASPECTS = spec.get("aspects", [])
    SEEDS = spec.get("seeds", {})
    _aspect_lines = ["Aspects:"]
    for a in ASPECTS:
        aspect_name = a.get('name', 'unnamed')
        weight = float(a.get('weight', 1.0))
        prompts = a.get('prompts', [])
        _aspect_lines.append(f"- **Aspect: {aspect_name}** (w={weight:.2f})")
        for prompt in prompts:
            _aspect_lines.append(f"  + Prompt: {prompt}")
    
    # Add seeds display
    _aspect_lines.append("\nSeeds:")
    unigrams = SEEDS.get("unigrams", [])
    phrases = SEEDS.get("phrases", [])
    if unigrams:
        _aspect_lines.append(f"- Unigrams ({len(unigrams)}): {', '.join(unigrams[:10])}{'...' if len(unigrams) > 10 else ''}")
    if phrases:
        _aspect_lines.append(f"- Phrases ({len(phrases)}): {', '.join(phrases[:5])}{'...' if len(phrases) > 5 else ''}")
    
    _aspects_header = "\n".join(_aspect_lines)
    
    # Check if using fallback and show notification
    if "fallback" in gemini_log.lower() or "translation" in gemini_log.lower():
        st.warning("⚠️ **Using Fallback Translation Method**")
        st.info("🔍 **Note:** When using fallback translation, transcript rerank may not work optimally as it relies on Vietnamese seeds for better matching.")
    
    # Display Gemini call and aspects
    def update_log():
        with log_container.container():
            st.text(gemini_log)
            st.markdown(_aspects_header)
    
    update_log()

    # 1) CLIP rank (only if run_flag or cache already available)
    if not st.session_state.get('run_flag') and not st.session_state.get('clip_cache'):
        st.stop()
    clip_dirs = []
    if "b1" in DATASETS: clip_dirs.append(B1["clip_dir"])
    if "b2" in DATASETS: clip_dirs.append(B2["clip_dir"])
    print_msg = f"[Rank] DATASETS={DATASETS} | clip_dirs={clip_dirs} | window={WINDOW_SEC}s | skip={SKIP_SEC}s"
    # Build a cache key based on aspects + runtime filters to avoid recomputation when sorting
    try:
        cache_key = json.dumps({
            "aspects": ASPECTS,
            "datasets": sorted(list(DATASETS)),
            "window": WINDOW_SEC,
            "skip": SKIP_SEC,
            "exclude_series": exclude_series_list,
            "exclude_videos": exclude_videos_list,
        }, ensure_ascii=False, sort_keys=True)
    except Exception:
        cache_key = None

    cache_hit = bool(st.session_state.get("clip_cache") and st.session_state["clip_cache"].get("key") == cache_key)
    if cache_hit:
        buf2 = io.StringIO("[CACHE] Using cached CLIP results for interactive sorting\n")
        results = st.session_state["clip_cache"]["results"]
        aspect_names = st.session_state["clip_cache"].get("aspect_names", [])
        aspect_weights = st.session_state["clip_cache"].get("aspect_weights", [1.0]*len(aspect_names))
    else:
        with st.spinner("🎬 Đang chạy CLIP search..."):
            (rank_out, buf2) = run_with_logs(
                rank_videos_clip_window,
                ASPECTS,
                LIMIT_VIDEOS,
                WINDOW_SEC,
                SKIP_SEC,
                None,
                EXCLUDE_SER,
                None,
                None,
                None,
                None,
            )
        if rank_out is None:
            st.error("Ranking failed. See logs.")
            st.stop()
        (results, aspect_names, aspect_weights) = rank_out
        # Save cache for subsequent sidebar sorting reruns
        try:
            st.session_state["clip_cache"] = {
                "key": cache_key,
                "results": results,
                "aspect_names": aspect_names,
                "aspect_weights": aspect_weights,
            }
        except Exception:
            pass

    # Ensure cache exists immediately after auto-run (first enhance)
    if not st.session_state.get("clip_cache"):
        try:
            st.session_state["clip_cache"] = {
                "key": cache_key,
                "results": results,
                "aspect_names": aspect_names,
                "aspect_weights": aspect_weights,
            }
        except Exception:
            pass

    # Apply search filters to CLIP results
    results = apply_search_filters(results, exclude_series_list, exclude_videos_list)

    # Add CLIP rank results to logs
    def update_log():
        with log_container.container():
            st.text(gemini_log)
            st.markdown(_aspects_header)
            st.text(print_msg)
            st.text(buf2.getvalue() if buf2 else "")
    
    update_log()

    # Row 7: Show CLIP results only when rerank is disabled
    if not rerank_enabled:
        # Cache results for interactive sorting without re-running (preserve existing key)
        try:
            _cc = dict(st.session_state.get("clip_cache", {}))
            _cc.update({"results": results, "aspect_names": aspect_names})
            st.session_state["clip_cache"] = _cc
        except Exception:
            pass

        # Render Aspects Sort + Results (in the same container, sort UI above results)
        with clip_results_container.container():
            if results and aspect_names:
                with st.expander("↔️ Adjust Aspects (Only works with basic CLIP mode)", expanded=False):
                    # Sliders to adjust aspect weights (persistent during session)
                    weight_key = st.session_state.get("clip_cache", {}).get("key", "default")
                    if 'aspect_weight_overrides' not in st.session_state:
                        st.session_state['aspect_weight_overrides'] = {}
                    overrides_all = st.session_state['aspect_weight_overrides']
                    overrides = overrides_all.get(weight_key, {})

                    cols = st.columns(len(aspect_names)) if len(aspect_names) > 0 else [st]
                    new_weights = []
                    for idx, name in enumerate(aspect_names):
                        default_w = float(aspect_weights[idx]) if idx < len(aspect_weights) else 1.0
                        default_w = max(0.5, min(1.6, default_w))
                        with cols[idx % len(cols)]:
                            w = st.slider(
                                label=name,
                                min_value=0.5,
                                max_value=1.6,
                                value=float(overrides.get(idx, default_w)),
                                step=0.05,
                                key=f"aspect_weight_slider_{idx}"
                            )
                            new_weights.append(w)
                    overrides_all[weight_key] = {i: new_weights[i] for i in range(len(new_weights))}
                    st.caption("Kéo slider để tăng/giảm trọng số từng aspect (0.5 → 1.6). Sắp xếp theo tổng điểm có trọng số mới.")

                    # Re-rank view by weighted sum of aspect scores using sliders (no rescoring)
                    def _weighted_sum(win, wts):
                        scores = win.get("aspect_scores", [])
                        s = 0.0
                        for i, w in enumerate(wts):
                            if i < len(scores):
                                s += float(w) * float(scores[i])
                        return s
                    results = sorted(results, key=lambda r: _weighted_sum(r["win"], new_weights), reverse=True)

            st.subheader("CLIP Multi-Aspects Results")
            if not results:
                st.info("No results found (check feature directories and dataset selection).")
            else:
                rows = min(TOP_VIDEOS, len(results))
                for i in range(rows):
                    r = results[i]
                    vid, win = r["video_id"], r["win"]
                    t0s, t1s = sec_to_mmss(win['t0']), sec_to_mmss(win['t1'])
                    with st.container():
                        st.markdown(f"**[{i+1}] {vid}** · total={r['total']:.3f} · win: {t0s} → {t1s}")
                        cols = st.columns(len(aspect_names))
                        for j, col in enumerate(cols):
                            with col:
                                frame = int(win["aspect_frames"][j])
                                score = float(win["aspect_scores"][j])
                                img = safe_imread(keyframe_path(vid, frame))
                                st.image(img, caption="")
                                st.markdown(f"**{aspect_names[j]}**")
                                st.markdown(f"frame={frame} s={score:.3f}")
                    url = get_watch_url(vid)
                    with st.popover("🎥 Watch Video", width='stretch'):
                        st.markdown(create_popover_css(), unsafe_allow_html=True)
                        col1, col2 = st.columns([2, 1]) # Căn lề radio qua phải
                        with col1:
                            st.markdown("")  # Cột trống để đẩy sang phải
                        with col2:
                            # Danh sách các lựa chọn (giữ logic đơn giản)
                            options = ("Youtube Online", "Kaggle Offline")
                            
                            # Hàm để định dạng cách hiển thị (thêm icon)
                            def format_func(option):
                                if option == "Youtube Online":
                                    return f"▶️ {option}"
                                elif option == "Kaggle Offline":
                                    return f"☁️ {option}"
                                return option

                            watch_mode = st.radio(
                                "Watch Mode Select",
                                options,
                                key=f"watch_mode_clip_{vid}",
                                horizontal=True,
                                label_visibility="collapsed",
                                format_func=format_func
                            )

                        start_sec = int(max(0, win['t0']))
                        actual_video_id = vid
                        actual_fps = get_fps_from_csv(actual_video_id)
                        
                        if watch_mode == "Youtube Online":
                            url = get_watch_url(vid)
                            if url:
                                youtube_video_id = url.split('v=')[1].split('&')[0] if 'v=' in url else url.split('/')[-1]
                                components.html(create_youtube_player_html(youtube_video_id, start_sec, fps=actual_fps, video_name=actual_video_id), height=1500)
                            else:
                                st.warning("YouTube URL not found for this video. Try Kaggle Offline mode.")
                        
                        elif watch_mode == "Kaggle Offline":
                            # Create placeholders for the progress UI
                            status_placeholder = st.empty()
                            
                            local_path = download_video_from_kaggle(actual_video_id, LOCAL_VIDEO_ROOT, status_placeholder)
                            
                            if local_path:
                                file_url = f"{LOCAL_VIDEO_URL_BASE}/{local_path.name}"
                                components.html(create_local_player_html(file_url, start_sec, fps=actual_fps, video_name=actual_video_id), height=1500, scrolling=True)
                        else:
                            # Error message is already shown by the download function
                            pass
                    st.divider()
        # Ensure rerank section is empty when disabled
        rerank_results_container.empty()

    # 2) Transcript rerank (only if enabled)
    if rerank_enabled:
        # Do not render initial CLIP grid; proceed directly to reranked results
        clip_results_container.empty()
        # Add rerank start message to logs
        def update_log():
            with log_container.container():
                st.text(gemini_log)
                st.markdown(_aspects_header)
                st.text(print_msg)
                st.text(buf2.getvalue() if buf2 else "")
                st.text("Running rerank_with_transcript_ultra_window() ...")
        
        update_log()
        with st.spinner("📑 Đang rerank theo transcript..."):
            (reranked_df, buf3) = run_with_logs(
                rerank_with_transcript_ultra_window,
                results, query_vi, SEEDS,
                60, DELTA_SEC,
                0.6, 0.4,
                {"any": 0.5, "phr": 0.5}, True, True,
                workers=8, phrases_only=False, max_unigrams=16, max_phrases=10,
                num_aspects=len(ASPECTS)
            )
        if reranked_df is None or isinstance(reranked_df, tuple):
            st.error("Rerank failed. See logs.")
            st.stop()
        # Add final rerank results to logs
        def update_log():
            with log_container.container():
                st.text(gemini_log)
                st.markdown(_aspects_header)
                st.text(print_msg)
                st.text(buf2.getvalue() if buf2 else "")
                st.text("Running rerank_with_transcript_ultra_window() ...")
                st.text(buf3.getvalue() if buf3 else "")
        
        update_log()
        # Show a pretty table for the rerank head inside the Logs expander
        with log_exp:
            try:
                _tbl = reranked_df.head(10).copy()
                _tbl["t0"] = [sec_to_mmss(t) for t in _tbl["t0"].values]
                _tbl["t1"] = [sec_to_mmss(t) for t in _tbl["t1"].values]
                _show_cols = ["video_id","clip_total","hits_any","hits_phr","clip_n","text_score","fused","t0","t1"]
                st.dataframe(_tbl[_show_cols].reset_index(drop=True), width='stretch')
            except Exception:
                pass

        top_after = reorder_results_by_df(results, reranked_df, top_videos=TOP_VIDEOS)

        # Row 8: Show reranked results
        with rerank_results_container.container():
            st.subheader("CLIP Multi-Aspects + Rerank with Transcript")
            if reranked_df.empty:
                st.info("No candidates for rerank.")
            else:
                rows = min(TOP_VIDEOS, len(top_after))
                for i in range(rows):
                    r = top_after[i]
                    vid, win = r["video_id"], r["win"]
                    t0s, t1s = sec_to_mmss(win['t0']), sec_to_mmss(win['t1'])
                    with st.container():
                        st.markdown(f"**[{i+1}] {vid}** · total={r['total']:.3f} · win: {t0s} → {t1s}")
                        cols = st.columns(len(aspect_names))
                        for j, col in enumerate(cols):
                            with col:
                                frame = int(win["aspect_frames"][j])
                                score = float(win["aspect_scores"][j])
                                img = safe_imread(keyframe_path(vid, frame))
                                st.image(img, caption="")
                                st.markdown(f"**{aspect_names[j]}**")
                                st.markdown(f"frame={frame} s={score:.3f}")
                    url = get_watch_url(vid)
                    if url:
                        # Watch Video button with popover
                        with st.popover("🎥 Watch Video", width='stretch'):
                            st.markdown(create_popover_css(), unsafe_allow_html=True)
                            col1, col2 = st.columns([2, 1]) # Căn lề radio qua phải
                            with col1:
                                st.markdown("")  # Cột trống để đẩy sang phải
                            with col2:
                                # Danh sách các lựa chọn (giữ logic đơn giản)
                                options = ("Youtube Online", "Kaggle Offline")
                                
                                # Hàm để định dạng cách hiển thị (thêm icon)
                                def format_func(option):
                                    if option == "Youtube Online":
                                        return f"▶️ {option}"
                                    elif option == "Kaggle Offline":
                                        return f"☁️ {option}"
                                    return option

                                watch_mode = st.radio(
                                    "Watch Mode Select",
                                    options,
                                    key=f"watch_mode_clip2_{vid}",
                                    horizontal=True,
                                    label_visibility="collapsed",
                                    format_func=format_func
                                )
                            # Use the actual video ID from dataset (not from YouTube URL)
                            start_sec = int(max(0, win['t0']))
                            actual_video_id = vid  # Use the dataset video ID (e.g., K16_V013)
                            youtube_video_id = url.split('v=')[1].split('&')[0] if 'v=' in url else url.split('/')[-1]
                            
                            # Get actual FPS from CSV file using dataset video ID
                            actual_fps = get_fps_from_csv(actual_video_id)
                            
                            # Use the working YouTube API with FPS-based frame calculation
                            components.html(create_youtube_player_html(youtube_video_id, start_sec, fps=actual_fps, video_name=actual_video_id), height=1500)
                    else:
                        st.markdown("No video URL available")
                    st.divider()