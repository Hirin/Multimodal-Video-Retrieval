"""
Multimodal Video Retrieval Application

A Streamlit-based video retrieval system using CLIP embeddings and BM25 text search.
Refactored to use modular architecture with services, utilities, and components.
"""

import os
import io
import json
import time
import shutil
import concurrent.futures
from pathlib import Path
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import streamlit.components.v1 as components

# Import configuration
from config import settings

# Import models
from models.video import SearchResult, VideoMetadata

# Import services
from services.clip_service import (
    ensure_clip_text_encoder,
    encode_text as clip_encode_text,
    compute_similarity
)
from services.text_search import (
    bm25_search_videos,
    deduplicate_bm25_results,
    get_transcript_snippets_with_highlights
)
from services.gemini_service import (
    extract_aspects_and_seeds_with_gemini,
    create_fallback_enhanced_data
)
from services.fallback_service import fallback_translate_and_create_aspects
from services.video_service import (
    keyframe_path,
    npy_path_for_video,
    map_csv_path_for_video,
    keep_idxs_path_for_video,
    safe_imread,
    get_fps_from_csv,
    timestamp_to_keyframe_number,
    load_times_for_video,
    load_video_features,
    list_videos_from_dirs
)
from services.database import (
    convert_bm25_syntax,
    highlight_keywords_in_text,
    count_matches,
    build_bm25_query_from_seeds
)
from services.server import make_video_app, ensure_server_once

# Import utilities
from utils.helpers import (
    l2_normalize_rows,
    sec_to_mmss,
    parse_mmss,
    dataset_for_video
)
from utils.session_state import SessionStateManager
from utils.logger import get_logger

# Import components
from components.styling import (
    create_popover_css,
    create_loading_overlay,
    create_dialog_loading_html,
    create_success_html,
    create_error_html
)

# Initialize logger
logger = get_logger('app')

# Initialize session state
SessionStateManager.initialize()

# ============================ KAGGLE DOWNLOADER ============================

def reset_watch_mode_keys():
    """Reset all watch mode keys to default (Youtube Online)"""
    if not SessionStateManager.is_dialog_active():
        SessionStateManager.clear_watch_modes()


@st.dialog("Đang tải video...")
def download_video_dialog(video_id: str, local_video_root: Path):
    """Show download dialog with loading animation and run download logic inside"""

    # Set dialog context to prevent watch mode reset
    SessionStateManager.set_dialog_context(True)

    # Create placeholder for dynamic content
    content_placeholder = st.empty()

    # Show loading animation
    loading_message = f"File chưa có sẵn. Đang tải {video_id}.mp4 từ Kaggle..."
    content_placeholder.markdown(create_dialog_loading_html(loading_message), unsafe_allow_html=True)

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

        logger.info(f"[KaggleHub] Downloading {remote_file_path} from {dataset_handle}...")

        # Download to system cache
        cached_path_str = kagglehub.dataset_download(dataset_handle, path=remote_file_path)
        cached_path = Path(cached_path_str)
        logger.info(f"[KaggleHub] Downloaded to cache: {cached_path}")

        # Move the file from cache to our local ./videos directory
        local_video_path = local_video_root / f"{video_id}.mp4"
        shutil.move(cached_path, local_video_path)
        logger.info(f"[KaggleHub] Moved to: {local_video_path}")

        # Show success message
        content_placeholder.markdown(create_success_html(), unsafe_allow_html=True)

        # Wait a bit then close dialog
        time.sleep(2.0)
        # Clear dialog context before rerun
        SessionStateManager.set_dialog_context(False)
        st.rerun()  # Refresh the main UI to show the video player
        return local_video_path

    except Exception as e:
        error_message = str(e)
        logger.error(f"[KaggleHub Error] {error_message}")

        # Show error message
        content_placeholder.markdown(create_error_html(), unsafe_allow_html=True)

        # Wait a bit then close dialog
        time.sleep(3.0)
        # Clear dialog context before rerun
        SessionStateManager.set_dialog_context(False)
        st.rerun()  # Refresh the main UI
        return None


def download_video_from_kaggle(video_id: str, local_video_root: Path, placeholder) -> Optional[Path]:
    """
    Downloads a video using the reliable kagglehub library, moves it from cache to the
    local videos folder, and shows a user-friendly dialog on the UI during the process.
    """
    local_video_path = local_video_root / f"{video_id}.mp4"

    # Debug info
    logger.debug(f"[Download] Checking video: {video_id}")
    logger.debug(f"[Download] Local path: {local_video_path}")
    logger.debug(f"[Download] Path exists: {local_video_path.exists()}")

    if local_video_path.exists():
        logger.debug(f"[Download] Video already exists, returning path")
        placeholder.empty()
        return local_video_path

    logger.info(f"[Download] Video not found, starting download dialog")
    # Show download dialog that runs download logic
    result = download_video_dialog(video_id, local_video_root)

    logger.debug(f"[Download] Dialog result: {result}")
    # Return the path if it exists after download
    if result and result.exists():
        logger.info(f"[Download] Download successful, returning path")
        return result
    else:
        logger.warning(f"[Download] Download failed or no result")
        return None


# ============================ PARQUET KEYFRAMES DATA ============================

@st.cache_data
def get_keyframes_data():
    """Load and cache keyframes data from parquet files"""
    data = {}
    try:
        # B1 keyframes
        b1_parquet = f"{settings.DATA_ROOT}b1/keyframes-info.parquet"
        if os.path.exists(b1_parquet):
            data['b1'] = pd.read_parquet(b1_parquet)
            logger.info(f"Loaded B1 keyframes data: {len(data['b1'])} rows")

        # B2 keyframes
        b2_parquet = f"{settings.DATA_ROOT}b2/keyframes-info.parquet"
        if os.path.exists(b2_parquet):
            data['b2'] = pd.read_parquet(b2_parquet)
            logger.info(f"Loaded B2 keyframes data: {len(data['b2'])} rows")
    except Exception as e:
        logger.error(f"Error loading keyframes data: {e}")

    return data


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

                        # This is a valid canonical frame
                        kf_path = keyframe_path(vid, frame)
                        if os.path.exists(kf_path):
                            return (frame, kf_path)
            except Exception as e:
                logger.debug(f"Error checking frame {frame}: {e}")
                continue

    # Fallback: use target_frame even if not in parquet
    kf_path = keyframe_path(vid, target_frame)
    return (target_frame, kf_path)


def get_fps_from_csv_safe(vid: str) -> float:
    """Safely get FPS from CSV, with fallback"""
    fps = get_fps_from_csv(vid)
    return fps if fps else 25.0


# ============================ VIDEO LISTING ============================

@st.cache_data
def cached_list_videos():
    """Cache the list of videos from both batches"""
    clip_dirs = []
    if "b1" in settings.DATASETS:
        clip_dirs.append(settings.B1["clip_dir"])
    if "b2" in settings.DATASETS:
        clip_dirs.append(settings.B2["clip_dir"])

    vids = list_videos_from_dirs(
        clip_dirs,
        limit_videos=settings.LIMIT_VIDEOS,
        exclude_ser=settings.EXCLUDE_SER
    )
    logger.info(f"Loaded {len(vids)} videos from cache")
    return vids


# ============================ CLIP SEARCH FUNCTIONS ============================

def multi_query_search(query_texts: List[str], weight_list: List[float], skip_sec: float,
                      window_sec: float, top_k: int, video_filter_list: Optional[List[str]] = None):
    """
    Multi-query CLIP search with weighted aspect queries.

    Args:
        query_texts: List of text queries (aspects)
        weight_list: Weight for each query
        skip_sec: Seconds to skip at the beginning
        window_sec: Time window size
        top_k: Number of top results to return
        video_filter_list: Optional list of videos to filter

    Returns:
        List of search results with aggregated scores
    """
    ensure_clip_text_encoder()

    # Encode all queries at once
    query_embeds = clip_encode_text(query_texts)
    query_embeds = l2_normalize_rows(query_embeds)

    # Get video list
    vids = cached_list_videos()
    if video_filter_list:
        vids = [v for v in vids if v in video_filter_list]

    results = []

    for vid in vids:
        try:
            # Load video features
            data = load_video_features(vid)
            if data is None:
                continue

            feats, times = data
            feats = l2_normalize_rows(feats)

            # Skip initial frames
            if skip_sec > 0:
                mask = times >= skip_sec
                feats = feats[mask]
                times = times[mask]

            if len(feats) == 0:
                continue

            # Compute similarities for all aspects
            all_sims = []
            for q_embed in query_embeds:
                sim = compute_similarity(
                    q_embed.reshape(1, -1),
                    feats,
                    normalize=False
                ).flatten()
                all_sims.append(sim)

            # Aggregate similarities with weights
            total_sim = np.zeros(len(feats))
            for sim, weight in zip(all_sims, weight_list):
                total_sim += sim * weight

            # Find best windows
            if window_sec > 0:
                # Window-based scoring
                for i in range(len(feats)):
                    t_start = times[i]
                    t_end = t_start + window_sec
                    mask = (times >= t_start) & (times <= t_end)
                    window_scores = total_sim[mask]

                    if len(window_scores) > 0:
                        score = float(np.mean(window_scores))
                        best_idx = i + np.argmax(window_scores)

                        results.append({
                            'video_id': vid,
                            'frame_id': int(best_idx),
                            'timestamp': float(times[i]),
                            'score': score,
                            'window_start': float(t_start),
                            'window_end': float(t_end)
                        })
            else:
                # Frame-level scoring
                for i in range(len(feats)):
                    results.append({
                        'video_id': vid,
                        'frame_id': i,
                        'timestamp': float(times[i]),
                        'score': float(total_sim[i])
                    })

        except Exception as e:
            logger.error(f"Error processing video {vid}: {e}")
            continue

    # Sort by score and return top-k
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]


# ============================ RERANK FUNCTIONS ============================

def build_keyword_sets(seeds: Dict, max_unigrams: int = 16, max_phrases: int = 10,
                       phrases_only: bool = False):
    """Build keyword sets from seeds for BM25 queries"""
    unigrams = seeds.get("unigrams", [])[:max_unigrams] if not phrases_only else []
    phrases = seeds.get("phrases", [])[:max_phrases]

    q_any_parts = list(unigrams) if unigrams else []
    q_phr_parts = [f'"{p}"' for p in phrases] if phrases else []

    q_any = " OR ".join(q_any_parts) if q_any_parts else ""
    q_phr = " OR ".join(q_phr_parts) if q_phr_parts else ""

    return (q_any, q_phr)


def rerank_with_transcript_ultra_window(
    results: List[Dict],
    query_vi: str,
    seeds: Dict,
    top_candidates: int,
    delta_sec: float,
    w_clip: float,
    w_text: float,
    text_weights: Dict[str, float],
    verbose: bool = True,
    sort_output: bool = True,
    workers: int = 8,
    phrases_only: bool = False,
    max_unigrams: int = 16,
    max_phrases: int = 10,
    num_aspects: int = 1
):
    """
    Rerank CLIP results using transcript matching with ultra window approach.
    """
    if not results:
        return pd.DataFrame(), io.StringIO()

    # Take top candidates
    top_results = results[:top_candidates]

    # Build BM25 queries from seeds
    q_any, q_phr = build_keyword_sets(seeds, max_unigrams, max_phrases, phrases_only)

    if not q_any and not q_phr:
        logger.warning("No valid keywords for rerank")
        return pd.DataFrame(), io.StringIO()

    # Prepare windows for each result
    windows = []
    for r in top_results:
        vid = r['video_id']
        t_center = r.get('timestamp', 0.0)

        # Create window around timestamp
        t0 = max(0.0, t_center - delta_sec / 2)
        t1 = t_center + delta_sec / 2

        windows.append({
            'video_id': vid,
            'frame_id': r.get('frame_id', 0),
            't0': t0,
            't1': t1,
            'clip_score': r.get('score', 0.0)
        })

    # Count transcript matches in parallel
    def count_window(win):
        vid = win['video_id']
        dataset = dataset_for_video(vid)
        db_path = settings.B1_DB_PATH if dataset == "b1" else settings.B2_DB_PATH

        if not db_path or not os.path.exists(db_path):
            return (vid, 0, 0)

        return count_matches(db_path, vid, win['t0'], win['t1'], q_any, q_phr)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        counts = list(executor.map(count_window, windows))

    # Build result dataframe
    rows = []
    for win, (_, h_any, h_phr) in zip(windows, counts):
        text_score = text_weights.get('any', 0.3) * h_any + text_weights.get('phr', 0.7) * h_phr
        clip_total = win['clip_score']
        clip_normalized = clip_total / max(1, num_aspects)
        fused = w_clip * clip_normalized + w_text * text_score

        rows.append({
            'video_id': win['video_id'],
            'frame_id': win['frame_id'],
            't0': win['t0'],
            't1': win['t1'],
            'clip_total': clip_total,
            'clip_n': clip_normalized,
            'hits_any': h_any,
            'hits_phr': h_phr,
            'text_score': text_score,
            'fused': fused
        })

    df = pd.DataFrame(rows)

    if sort_output and not df.empty:
        df = df.sort_values('fused', ascending=False).reset_index(drop=True)

    buf = io.StringIO()
    buf.write(f"Reranked {len(df)} windows\n")
    buf.write(f"Query keywords: any={len(q_any.split(' OR '))}, phrase={len(q_phr.split(' OR '))}\n")

    return df, buf


def reorder_results_by_df(clip_results: List[Dict], rerank_df: pd.DataFrame, top_videos: int = 20):
    """Reorder CLIP results based on rerank dataframe"""
    if rerank_df.empty:
        return clip_results[:top_videos]

    # Build lookup by video_id
    video_scores = {}
    for _, row in rerank_df.iterrows():
        vid = row['video_id']
        if vid not in video_scores or row['fused'] > video_scores[vid]['score']:
            video_scores[vid] = {
                'score': row['fused'],
                'window': {
                    't0': row['t0'],
                    't1': row['t1'],
                    'frame_id': row['frame_id']
                }
            }

    # Reorder results
    reordered = []
    for vid, data in sorted(video_scores.items(), key=lambda x: x[1]['score'], reverse=True):
        # Find original result
        orig = next((r for r in clip_results if r['video_id'] == vid), None)
        if orig:
            reordered.append({
                **orig,
                'total': data['score'],
                'win': data['window']
            })

    return reordered[:top_videos]


# ============================ UTILITY FUNCTIONS ============================

def run_with_logs(func, *args, **kwargs):
    """Run a function and capture its output to a StringIO buffer"""
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = buf = StringIO()

    try:
        result = func(*args, **kwargs)
        return result, buf
    finally:
        sys.stdout = old_stdout


# ============================ SIDEBAR CONFIGURATION ============================

with st.sidebar:
    st.title("Configuration")

    # Search Mode Selection
    MODES = [
        "CLIP Multi-aspects Search",
        "CLIP Multi-aspects Search + Rerank",
        "Transcript Seed Search (BM25)",
        "BM25 Keyword Search",
        "Search by Video ID",
    ]
    search_mode = st.selectbox("🔎 Search Mode", MODES, index=0, key="search_mode_sel")

    # Gemini availability toggle
    gemini_configured = bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip())
    turn_off_gemini = st.checkbox(
        "🔌 Turn off Gemini",
        value=False,
        key="turn_off_gemini",
        help="Disable Gemini API to skip Enhance step. Rerank requires Gemini and will be disabled."
    )
    gemini_available = gemini_configured and not turn_off_gemini

    # Filter options
    with st.expander("🔧 Search Filters", expanded=True):
        # Dataset selection
        st.markdown("**Dataset Selection:**")
        batch_label = st.selectbox("Choose Dataset:", ["Both Batches", "Batch 1", "Batch 2"], index=0, label_visibility="collapsed")

        st.markdown("**Advanced Filters:**")
        # Filter inputs
        exclude_series = st.text_input(
            "❌ Exclude Series:",
            placeholder="L01,L02,K01",
            help="Exclude video series (comma-separated). E.g., L01,L02 to exclude batch 1 videos",
            key="exclude_series"
        )
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
        active_datasets = {"b1"}
    elif batch_label == "Batch 2":
        active_datasets = {"b2"}
    else:
        active_datasets = {"b1", "b2"}

    # Update global settings (for this session)
    settings.DATASETS = active_datasets

    # Show active filters summary
    active_filters = []
    if batch_label != "Both Batches":
        active_filters.append(f"Dataset: {batch_label}")
    if exclude_series_list:
        active_filters.append(f"Exclude {len(exclude_series_list)} series")
    if exclude_videos_list:
        active_filters.append(f"Exclude {len(exclude_videos_list)} videos")

    if active_filters:
        st.success(f"**Active Filters:** {', '.join(active_filters)}")
    else:
        st.info("**No filters applied**")

    # Gemini API Configuration
    with st.expander("🔑 Gemini API Configuration", expanded=False):
        st.markdown("**Configure and test your Gemini API connection**")

        # API Key input
        api_key = st.text_input(
            "Gemini API Key:",
            value=settings.GEMINI_API_KEY,
            type="password",
            help="Enter your Google Gemini API key."
        )

        # Update the global API key if changed
        if api_key and api_key != settings.GEMINI_API_KEY:
            settings.GEMINI_API_KEY = api_key
            os.environ["GOOGLE_API_KEY"] = api_key

        # Gemini Configuration Options
        if api_key and api_key.strip():
            st.markdown("**Model Selection & Parameters:**")
            gemini_model = st.selectbox(
                "Model",
                ["gemini-2.5-flash", "gemini-2.5-pro"],
                index=0,
                help="Pro: High quality, Flash: Fast speed."
            )
            col_temp, col_topp = st.columns(2)
            gemini_temperature = col_temp.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
            gemini_top_p = col_topp.slider("Top P", 0.0, 1.0, 0.9, 0.05)
        else:
            gemini_model = "gemini-2.5-flash"
            gemini_temperature = 0.7
            gemini_top_p = 0.9

        test_button = st.button("🧪 Test Gemini API", type="primary")

        if test_button:
            if not api_key:
                st.error("❌ Please enter your Gemini API key first!")
            else:
                with st.spinner("Testing Gemini API connection..."):
                    try:
                        from google import genai
                        client = genai.Client(api_key=api_key)
                        resp = client.models.generate_content(
                            model='gemini-2.0-flash',
                            contents="Hello! Please respond with 'API connection successful' if you can read this message."
                        )
                        response_text = (getattr(resp, "text", None) or getattr(resp, "output_text", None) or "").strip()
                        if response_text:
                            st.success("✅ **Test Successful!**")
                            st.info(f"**Response:** {response_text}")
                        else:
                            st.warning("⚠️ **API responded but with empty content**")
                    except Exception as e:
                        st.error("❌ **Test Failed!**")
                        st.error(f"**Error:** {str(e)}")

        if settings.GEMINI_API_KEY:
            st.success("🔑 **API Key Status:** Configured")
        else:
            st.warning("⚠️ **API Key Status:** Not configured")

    # General Settings
    with st.expander("⚙️ General Settings", expanded=False):
        st.markdown("**Time & Ranking**")

        # Get current values or defaults
        _win_default = sec_to_mmss(SessionStateManager.get('WINDOW_SEC', settings.WINDOW_SEC))
        _skip_default = sec_to_mmss(SessionStateManager.get('SKIP_SEC', settings.SKIP_SEC))
        _del_default = sec_to_mmss(SessionStateManager.get('DELTA_SEC', settings.DELTA_SEC))
        _top_default = int(SessionStateManager.get('TOP_VIDEOS', settings.TOP_VIDEOS))

        win_str_inp = st.text_input("Window", value=_win_default, key="win_str_inp")
        skip_str_inp = st.text_input("Skip Intro", value=_skip_default, key="skip_str_inp")
        del_str_inp = st.text_input("Rerank Window", value=_del_default, key="del_str_inp")
        topk_inp = st.number_input("Top K", 1, 200, value=_top_default, step=1, key="topk_inp")

        pending_change = (
            sec_to_mmss(SessionStateManager.get('WINDOW_SEC', settings.WINDOW_SEC)) != win_str_inp or
            sec_to_mmss(SessionStateManager.get('SKIP_SEC', settings.SKIP_SEC)) != skip_str_inp or
            sec_to_mmss(SessionStateManager.get('DELTA_SEC', settings.DELTA_SEC)) != del_str_inp or
            int(SessionStateManager.get('TOP_VIDEOS', settings.TOP_VIDEOS)) != int(topk_inp)
        )

        col_upd, col_reset = st.columns(2)
        if pending_change:
            update_cfg = col_upd.button("Update", type="secondary", use_container_width=True)
            reset_cfg = col_reset.button("Reset", use_container_width=True)
        else:
            update_cfg = False
            reset_cfg = False

        default_cfg = st.button("Default", use_container_width=True)

        if update_cfg:
            SessionStateManager.set('WINDOW_SEC', parse_mmss(win_str_inp, settings.WINDOW_SEC))
            SessionStateManager.set('SKIP_SEC', parse_mmss(skip_str_inp, settings.SKIP_SEC))
            SessionStateManager.set('DELTA_SEC', parse_mmss(del_str_inp, settings.DELTA_SEC))
            SessionStateManager.set('TOP_VIDEOS', int(topk_inp))
            st.rerun()

        if reset_cfg:
            # Reset to current session values
            st.rerun()

        if default_cfg:
            SessionStateManager.set('WINDOW_SEC', 40.0)
            SessionStateManager.set('SKIP_SEC', 0.0)
            SessionStateManager.set('DELTA_SEC', 20.0)
            SessionStateManager.set('TOP_VIDEOS', 20)
            st.rerun()

    # Debug toggle
    debug_mode = st.checkbox("🐛 Debug Mode", value=SessionStateManager.get("debug_mode", False))
    SessionStateManager.set("debug_mode", debug_mode)

# ============================ SIDEBAR END =======================================

st.title("Video Retrieval - CLIP + BM25 Search")
st.markdown("*Use the sidebar to select a search mode and configure settings.*")

# Mode mapping
def _mode_key(name: str) -> str:
    return {
        "CLIP Multi-aspects Search"            : "clip",
        "CLIP Multi-aspects Search + Rerank"   : "rerank",
        "Transcript Seed Search (BM25)"        : "bm25_seed",
        "BM25 Keyword Search"                  : "bm25_kw",
        "Search by Video ID"                   : "vid",
    }[name]

active_key = _mode_key(search_mode)
clip_enabled        = (search_mode == "CLIP Multi-aspects Search")
rerank_enabled      = (search_mode == "CLIP Multi-aspects Search + Rerank")
bm25_seed_enabled   = (search_mode == "Transcript Seed Search (BM25)")
bm25_enabled        = (search_mode == "BM25 Keyword Search")
video_id_enabled    = (search_mode == "Search by Video ID")

# Run flag management
def _clear_run_flags(except_mode_key: str | None = None):
    for k in list(st.session_state.keys()):
        if k.startswith("run_flag_") and (except_mode_key is None or k != f"run_flag_{except_mode_key}"):
            SessionStateManager.delete(k)

def _get_run_flag(mode_key: str) -> bool:
    return SessionStateManager.get(f"run_flag_{mode_key}", False)

def _set_run_flag(mode_key: str, value: bool):
    SessionStateManager.set(f"run_flag_{mode_key}", bool(value))

# Reset run/caches when switching modes
if 'last_mode' not in st.session_state:
    SessionStateManager.set('last_mode', search_mode)
elif SessionStateManager.get('last_mode') != search_mode:
    SessionStateManager.set('run_flag', False)
    SessionStateManager.set('auto_run_after_enhance', False)
    SessionStateManager.set('last_mode', search_mode)

# Apply runtime overrides from session
WINDOW_SEC = SessionStateManager.get('WINDOW_SEC', settings.WINDOW_SEC)
SKIP_SEC = SessionStateManager.get('SKIP_SEC', settings.SKIP_SEC)
DELTA_SEC = SessionStateManager.get('DELTA_SEC', settings.DELTA_SEC)
TOP_VIDEOS = SessionStateManager.get('TOP_VIDEOS', settings.TOP_VIDEOS)

# Max aspects input
max_aspects_input = st.text_input(
    "🎯 Max aspects:",
    value="",
    placeholder="Không nhập thì ngẫu nhiên",
    help="Số lượng aspects Gemini tạo (để trống = tự động)",
    key="max_aspects_input"
)

# If Gemini is not available, prevent selecting rerank mode
if (not gemini_available) and rerank_enabled:
    st.warning("⚠️ Rerank requires Gemini. Switching to basic CLIP search.")
    search_mode = "CLIP Multi-aspects Search"
    rerank_enabled = False
    clip_enabled = True

# Rerank weight settings
if rerank_enabled:
    with st.expander("🔧 Rerank Weight Settings", expanded=False):
        st.markdown("**Rerank Weight Settings**")
        col_w_clip, col_w_text, col_w_any, col_w_phr = st.columns(4)
        w_clip_slider = col_w_clip.slider("w_clip", 0.0, 1.0, float(SessionStateManager.get('W_CLIP', 0.4)), 0.05, key="w_clip_slider")
        w_text_slider = col_w_text.slider("w_text", 0.0, 1.0, float(SessionStateManager.get('W_TEXT', 0.6)), 0.05, key="w_text_slider")
        w_any_slider = col_w_any.slider("w_any (unigram)", 0.0, 1.0, float(SessionStateManager.get('W_ANY', 0.3)), 0.05, key="w_any_slider")
        w_phr_slider = col_w_phr.slider("w_phr (phrase)", 0.0, 1.0, float(SessionStateManager.get('W_PHR', 0.7)), 0.05, key="w_phr_slider")

        current_w_clip = SessionStateManager.get('W_CLIP', 0.4)
        current_w_text = SessionStateManager.get('W_TEXT', 0.6)
        current_w_any = SessionStateManager.get('W_ANY', 0.3)
        current_w_phr = SessionStateManager.get('W_PHR', 0.7)

        st.info(f"**Current Formula:** `fused = {current_w_clip}*(clip_total/num_aspects) + {current_w_text}*({current_w_any}*hits_any + {current_w_phr}*hits_phr)`")

        weights_changed = (current_w_clip != w_clip_slider or current_w_text != w_text_slider or
                          current_w_any != w_any_slider or current_w_phr != w_phr_slider)
        if weights_changed:
            if st.button("Update Weights", type="secondary", use_container_width=True):
                SessionStateManager.set('W_CLIP', w_clip_slider)
                SessionStateManager.set('W_TEXT', w_text_slider)
                SessionStateManager.set('W_ANY', w_any_slider)
                SessionStateManager.set('W_PHR', w_phr_slider)
                st.success("✅ Rerank weight settings updated!")
                st.rerun()

# BM25 options
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

# Video ID input
if video_id_enabled:
    st.markdown("**Video ID Search:**")
    video_id_input = st.text_input(
        "Enter Video ID:",
        placeholder="e.g., K01_V001, L02_V005",
        help="Enter the video ID to search for",
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

# Query input
if not video_id_enabled:
    query_vi = st.text_area("Enter Query:", height=150, placeholder=placeholder_text)
else:
    query_vi = ""

# Button styling
st.markdown(create_popover_css(), unsafe_allow_html=True)
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

# Initialize query enhancement state
if 'query_enhanced' not in st.session_state:
    SessionStateManager.set('query_enhanced', False)
if 'enhanced_data' not in st.session_state:
    SessionStateManager.set('enhanced_data', None)

# ============================ SEARCH MODE HANDLERS ============================

# BM25 Keyword Search Mode
if bm25_enabled:
    run_button = st.button("🚀 Run Query", width='stretch', type="primary")
    if run_button:
        reset_watch_mode_keys()

        if not query_vi.strip():
            st.error("Please enter a query first!")
        else:
            _set_run_flag("bm25_kw", True)
            st.info("🔍 **BM25 Keyword Search** - Searching directly by keywords")

    if _get_run_flag("bm25_kw"):
        st.markdown("---")
        st.subheader("BM25 Search Results")

        with st.spinner("🔍 Searching transcripts..."):
            results = bm25_search_videos(query_vi, limit=50)

        if results:
            st.success(f"✅ Found {len(results)} results")

            # Show results
            for i, result in enumerate(results[:TOP_VIDEOS]):
                vid = result['video_id']
                timestamp = result['timestamp']
                score = result['score']
                highlighted_text = result.get('highlighted_full_text', result.get('full_text', ''))

                with st.container():
                    st.markdown(f"**[{i+1}] {vid}** · time: {sec_to_mmss(timestamp)} · score: {score:.3f}")

                    # Show highlighted transcript
                    if highlighted_text:
                        st.markdown(f'<div class="bm25-fulltext">{highlighted_text}</div>', unsafe_allow_html=True)

                    # Show keyframe
                    frame_num = timestamp_to_keyframe_number(vid, timestamp)
                    best_frame, kf_path = find_best_keyframe(vid, frame_num)

                    if os.path.exists(kf_path):
                        img = safe_imread(kf_path)
                        if img:
                            st.image(img, caption=f"{vid} - Frame {best_frame} @ {sec_to_mmss(timestamp)}", use_container_width=True)
        else:
            st.warning("No results found. Try a different query.")

# Video ID Search Mode
elif video_id_enabled:
    run_button = st.button("🔍 Search Video", width='stretch', type="primary")
    if run_button:
        reset_watch_mode_keys()

        if not video_id_input.strip():
            st.error("Please enter a video ID first!")
        else:
            _set_run_flag("vid", True)
            st.info("🔍 **Video ID Search** - Searching for specific video")

    if _get_run_flag("vid"):
        st.markdown("---")
        st.subheader("Video ID Search Results")

        vid = video_id_input.strip()

        # Check if video exists
        npy_file = npy_path_for_video(vid)
        if os.path.exists(npy_file):
            st.success(f"✅ Found video: {vid}")

            # Load video features
            data = load_video_features(vid)
            if data:
                feats, times = data
                st.info(f"Video has {len(feats)} keyframes, duration: ~{sec_to_mmss(times[-1])}")

                # Show some keyframes
                st.markdown("**Sample Keyframes:**")
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    if i < len(times):
                        frame_num = timestamp_to_keyframe_number(vid, times[i * len(times) // 5])
                        best_frame, kf_path = find_best_keyframe(vid, frame_num)

                        if os.path.exists(kf_path):
                            img = safe_imread(kf_path)
                            if img:
                                with col:
                                    st.image(img, caption=f"{sec_to_mmss(times[i * len(times) // 5])}", use_container_width=True)
        else:
            st.error(f"❌ Video '{vid}' not found in database.")

# CLIP Multi-aspects Search (with or without Rerank)
elif clip_enabled or rerank_enabled:
    # Enhancement step
    if gemini_available:
        if not SessionStateManager.get('query_enhanced', False):
            enhance_button = st.button("🌱 Enhance Query", type="primary", width='stretch')
            if enhance_button:
                _clear_run_flags()
                if not query_vi.strip():
                    st.error("Please enter a query first!")
                else:
                    with st.spinner("Enhancing query with Gemini..."):
                        try:
                            enhanced_data = extract_aspects_and_seeds_with_gemini(
                                query_vi,
                                gemini_temperature,
                                gemini_top_p,
                                gemini_model,
                                max_aspects_input.strip() if max_aspects_input.strip() else "N"
                            )
                            SessionStateManager.set('enhanced_data', enhanced_data)
                            SessionStateManager.set('query_enhanced', True)
                            SessionStateManager.set('auto_run_after_enhance', True)
                            st.success("✅ Query enhanced successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Enhancement failed: {e}")
        else:
            # Show Run/Reset buttons
            col1, col2 = st.columns(2)
            with col1:
                run_button = st.button("🚀 Run Search", type="primary", use_container_width=True)
                if run_button:
                    reset_watch_mode_keys()
                    _set_run_flag(active_key, True)

            with col2:
                reset_button = st.button("🔄 Reset", use_container_width=True)
                if reset_button:
                    SessionStateManager.set('query_enhanced', False)
                    SessionStateManager.set('enhanced_data', None)
                    _clear_run_flags()
                    st.rerun()
    else:
        # No Gemini - direct run
        run_button = st.button("🚀 Run Search", type="primary", width='stretch')
        if run_button:
            reset_watch_mode_keys()
            if not query_vi.strip():
                st.error("Please enter a query first!")
            else:
                # Use fallback enhancement
                with st.spinner("Preparing search..."):
                    enhanced_data = create_fallback_enhanced_data(query_vi)
                    SessionStateManager.set('enhanced_data', enhanced_data)
                _set_run_flag(active_key, True)

    # Execute search if run flag is set
    if _get_run_flag(active_key):
        st.markdown("---")

        enhanced_data = SessionStateManager.get('enhanced_data')
        if not enhanced_data:
            st.error("No enhanced data available. Please enhance query first.")
            st.stop()

        ASPECTS = enhanced_data.get('aspects', [])
        SEEDS = enhanced_data.get('seeds', {})

        if not ASPECTS:
            st.error("No aspects found in enhanced data.")
            st.stop()

        # Display aspects
        st.subheader("Search Aspects")
        aspect_names = [a['name'] for a in ASPECTS]
        for i, aspect in enumerate(ASPECTS):
            with st.expander(f"**{i+1}. {aspect['name']}** (weight: {aspect.get('weight', 1.0)})"):
                st.write("**Prompts:**")
                for prompt in aspect.get('prompts', []):
                    st.write(f"- {prompt}")

        # Run CLIP search
        with st.spinner("🔍 Searching with CLIP..."):
            query_texts = []
            weights = []
            for aspect in ASPECTS:
                for prompt in aspect.get('prompts', []):
                    query_texts.append(prompt)
                    weights.append(aspect.get('weight', 1.0))

            results = multi_query_search(
                query_texts,
                weights,
                SKIP_SEC,
                WINDOW_SEC,
                TOP_VIDEOS * 3,  # Get more candidates for reranking
                exclude_videos_list if exclude_videos_list else None
            )

        if not results:
            st.warning("No results found.")
            st.stop()

        st.success(f"✅ Found {len(results)} results")

        # Rerank if enabled
        if rerank_enabled:
            st.markdown("---")
            st.subheader("Reranking with Transcript")

            with st.spinner("📑 Reranking results..."):
                w_clip = SessionStateManager.get('W_CLIP', 0.4)
                w_text = SessionStateManager.get('W_TEXT', 0.6)
                w_any = SessionStateManager.get('W_ANY', 0.3)
                w_phr = SessionStateManager.get('W_PHR', 0.7)

                reranked_df, buf = rerank_with_transcript_ultra_window(
                    results,
                    query_vi,
                    SEEDS,
                    min(60, len(results)),
                    DELTA_SEC,
                    w_clip,
                    w_text,
                    {'any': w_any, 'phr': w_phr},
                    verbose=True,
                    sort_output=True,
                    workers=8,
                    phrases_only=False,
                    max_unigrams=16,
                    max_phrases=10,
                    num_aspects=len(ASPECTS)
                )

            if not reranked_df.empty:
                st.success(f"✅ Reranked {len(reranked_df)} results")

                # Reorder results
                final_results = reorder_results_by_df(results, reranked_df, TOP_VIDEOS)

                # Display reranked results
                for i, r in enumerate(final_results[:TOP_VIDEOS]):
                    vid = r['video_id']
                    win = r.get('win', {})
                    t0 = win.get('t0', r.get('timestamp', 0))
                    t1 = win.get('t1', t0 + WINDOW_SEC)

                    with st.container():
                        st.markdown(f"**[{i+1}] {vid}** · score: {r.get('total', 0):.3f} · window: {sec_to_mmss(t0)} → {sec_to_mmss(t1)}")

                        # Show keyframe
                        frame_num = timestamp_to_keyframe_number(vid, t0)
                        best_frame, kf_path = find_best_keyframe(vid, frame_num)

                        if os.path.exists(kf_path):
                            img = safe_imread(kf_path)
                            if img:
                                st.image(img, caption=f"{vid} @ {sec_to_mmss(t0)}", use_container_width=True)
            else:
                st.warning("Reranking produced no results.")

        else:
            # Display CLIP results without reranking
            st.subheader("CLIP Search Results")

            for i, r in enumerate(results[:TOP_VIDEOS]):
                vid = r['video_id']
                timestamp = r.get('timestamp', 0)
                score = r.get('score', 0)

                with st.container():
                    st.markdown(f"**[{i+1}] {vid}** · score: {score:.3f} · time: {sec_to_mmss(timestamp)}")

                    # Show keyframe
                    frame_num = timestamp_to_keyframe_number(vid, timestamp)
                    best_frame, kf_path = find_best_keyframe(vid, frame_num)

                    if os.path.exists(kf_path):
                        img = safe_imread(kf_path)
                        if img:
                            st.image(img, caption=f"{vid} @ {sec_to_mmss(timestamp)}", use_container_width=True)

# Transcript Seed Search (BM25)
elif bm25_seed_enabled:
    if gemini_available:
        if not SessionStateManager.get('query_enhanced', False):
            enhance_button = st.button("🌱 Enhance Query", type="primary", width='stretch')
            if enhance_button:
                _clear_run_flags()
                if not query_vi.strip():
                    st.error("Please enter a query first!")
                else:
                    with st.spinner("Enhancing query with Gemini..."):
                        try:
                            enhanced_data = extract_aspects_and_seeds_with_gemini(
                                query_vi,
                                gemini_temperature,
                                gemini_top_p,
                                gemini_model,
                                max_aspects_input.strip() if max_aspects_input.strip() else "N"
                            )
                            SessionStateManager.set('enhanced_data', enhanced_data)
                            SessionStateManager.set('query_enhanced', True)
                            SessionStateManager.set('auto_run_after_enhance', True)
                            st.success("✅ Query enhanced successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Enhancement failed: {e}")
        else:
            # Show Run/Reset buttons
            col1, col2 = st.columns(2)
            with col1:
                run_button = st.button("🚀 Run Search", type="primary", use_container_width=True)
                if run_button:
                    reset_watch_mode_keys()
                    _set_run_flag("bm25_seed", True)

            with col2:
                reset_button = st.button("🔄 Reset", use_container_width=True)
                if reset_button:
                    SessionStateManager.set('query_enhanced', False)
                    SessionStateManager.set('enhanced_data', None)
                    _clear_run_flags()
                    st.rerun()
    else:
        # No Gemini - use manual seed input
        st.warning("⚠️ Gemini not available. Enter seeds manually or enable Gemini.")

        seeds_unigrams = st.text_input("Unigrams (comma-separated):", placeholder="cát, điêu, khắc")
        seeds_phrases = st.text_input("Phrases (comma-separated):", placeholder="điêu khắc cát, nghệ thuật")

        run_button = st.button("🚀 Run Search", type="primary", width='stretch')
        if run_button:
            reset_watch_mode_keys()

            # Build seeds from manual input
            unigrams = [s.strip() for s in seeds_unigrams.split(',') if s.strip()]
            phrases = [s.strip() for s in seeds_phrases.split(',') if s.strip()]

            if not unigrams and not phrases:
                st.error("Please enter at least some keywords!")
            else:
                enhanced_data = {
                    'seeds': {
                        'unigrams': unigrams,
                        'phrases': phrases
                    }
                }
                SessionStateManager.set('enhanced_data', enhanced_data)
                _set_run_flag("bm25_seed", True)

    # Execute search if run flag is set
    if _get_run_flag("bm25_seed"):
        st.markdown("---")

        enhanced_data = SessionStateManager.get('enhanced_data')
        if not enhanced_data:
            st.error("No enhanced data available.")
            st.stop()

        SEEDS = enhanced_data.get('seeds', {})

        # Display seeds
        st.subheader("Search Seeds")
        with st.expander("**Keywords**", expanded=True):
            st.write("**Unigrams:**", ", ".join(SEEDS.get('unigrams', [])))
            st.write("**Phrases:**", ", ".join(SEEDS.get('phrases', [])))

        # Build BM25 query
        bm25_query = build_bm25_query_from_seeds(query_vi, SEEDS)

        if not bm25_query:
            st.error("No valid keywords to search.")
            st.stop()

        with st.spinner("🔍 Searching transcripts with seeds..."):
            results = bm25_search_videos(bm25_query, limit=50)

        if results:
            st.success(f"✅ Found {len(results)} results")

            # Deduplicate results
            results = deduplicate_bm25_results(results, time_threshold=30.0)

            # Show results
            for i, result in enumerate(results[:TOP_VIDEOS]):
                vid = result['video_id']
                timestamp = result['timestamp']
                score = result['score']
                highlighted_text = result.get('highlighted_full_text', result.get('full_text', ''))

                with st.container():
                    st.markdown(f"**[{i+1}] {vid}** · time: {sec_to_mmss(timestamp)} · score: {score:.3f}")

                    # Show highlighted transcript
                    if highlighted_text:
                        st.markdown(f'<div class="bm25-fulltext">{highlighted_text}</div>', unsafe_allow_html=True)

                    # Show keyframe
                    frame_num = timestamp_to_keyframe_number(vid, timestamp)
                    best_frame, kf_path = find_best_keyframe(vid, frame_num)

                    if os.path.exists(kf_path):
                        img = safe_imread(kf_path)
                        if img:
                            st.image(img, caption=f"{vid} - Frame {best_frame} @ {sec_to_mmss(timestamp)}", use_container_width=True)
        else:
            st.warning("No results found. Try different keywords.")

# ============================ VIDEO SERVER ============================

# Start local video server if needed
LOCAL_VIDEO_ROOT = settings.LOCAL_VIDEO_ROOT
LOCAL_VIDEO_URL_BASE = settings.LOCAL_VIDEO_URL_BASE

# Ensure video root exists
LOCAL_VIDEO_ROOT.mkdir(exist_ok=True)

# Start Flask server
video_app = make_video_app(str(LOCAL_VIDEO_ROOT))
ensure_server_once(video_app, settings.FLASK_HOST, settings.FLASK_PORT)

# ============================ FOOTER ============================

st.markdown("---")
st.markdown("**Multimodal Video Retrieval System** · Powered by CLIP + BM25 + Gemini")
