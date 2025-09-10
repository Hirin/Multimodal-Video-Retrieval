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

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import streamlit.components.v1 as components
from deep_translator import GoogleTranslator

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

    # Search recursively for .db files
    search_pattern = f"{data_root}{batch_name}/**/*.db"
    db_files = glob.glob(search_pattern, recursive=True)
    bm25_files = [f for f in db_files if 'bm25' in f.lower() or 'text' in f.lower()]

    if bm25_files:
        return bm25_files[0]

    return None

B1_DB_PATH = find_db_path(DATA_ROOT, "b1")
B2_DB_PATH = find_db_path(DATA_ROOT, "b2")

DATASETS = {"b1", "b2"}
GEMINI_MODEL = "gemini-2.5-pro"
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
    sys_inst = (
        "Bạn là công cụ xây dựng truy vấn cho video retrieval.\n"
        "- Đầu vào: miêu tả TỰ NHIÊN (tiếng Việt/Anh).\n"
        "- Đầu ra: JSON với hai trường: 'aspects' và 'seeds'.\n"
        f"  * aspects: mảng {max_aspects} phần tử. Mỗi phần tử có name(VI), prompts(2-4 EN), weight trong khoảng [0.8, 1.3](Tự đánh giá).\n"
        "  * seeds: { 'unigrams': ≤28, 'phrases': ≤18 } bằng TIẾNG VIỆT.\n"
        "JSON hợp lệ, không kèm bình luận."
    )
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
    max_unigrams: int = 16, max_phrases: int = 10
):
    if text_weights is None:
        text_weights = {"any": 0.5, "phr": 0.5}
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
    df["clip_n"] = _minmax(df["clip_total"].values)
    df["any_n"] = _minmax(df["hits_any"].values)
    df["phr_n"] = _minmax(df["hits_phr"].values)
    a, b = text_weights["any"], text_weights["phr"]
    df["text_score"] = a * df["any_n"] + b * df["phr_n"]
    df["fused"] = w_clip * df["clip_n"] + w_text * df["text_score"]
    df = df.sort_values("fused", ascending=False).reset_index(drop=True)
    if verbose:
        print(f"[ULTRA-WIN] fused = {w_clip}*clip + {w_text}*( {a}*hits_any + {b}*hits_phr ) | Δ={delta_sec}s | workers={workers} | phrases_only={phrases_only}")
        # Removed printing the full head table in logs to avoid duplication; table is rendered via Streamlit.
        # (kept summary line above)

    if timing:
        print(f"[ULTRA-WIN] elapsed: {time.perf_counter() - t0:.3f}s (candidates={len(df)})")
    return df

def reorder_results_by_df(orig_results, ranked_df, top_videos=TOP_VIDEOS):
    order = ranked_df.head(top_videos)["video_id"].tolist()
    vid2r = {r["video_id"]: r for r in orig_results}
    return [vid2r[v] for v in order if v in vid2r]

# =============================== MEDIA INFO (watch_url) ==================================

def create_youtube_player_html(video_id: str, start_time: float, fps: float = 25.0) -> str:
    """Create HTML with working YouTube API and FPS-based frame calculation"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YouTube Player</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 10px; }}
            .container {{ display: flex; gap: 20px; }}
            .info-panel {{ flex: 1; background: #2e2e2e; padding: 20px; border-radius: 8px; color: white; }}
            .video-panel {{ flex: 2; }}
            .time-display {{ font-size: 24px; color: #00ff00; margin: 10px 0; }}
            .frame-display {{ font-size: 20px; color: #ff6b6b; margin: 10px 0; }}
            iframe {{ width: 100%; height: 500px; }}
            .status {{ margin: 10px 0; }}
            .debug {{ background: #1a1a1a; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 12px; max-height: 150px; overflow-y: auto; }}
            .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
            .info-item {{ background: #3a3a3a; padding: 8px; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="info-panel">
                <h4>Current Time:</h4>
                <div class="time-display" id="current-time-{video_id}">{start_time:.1f}s</div>
                
                <h4>Current Frame:</h4>
                <div class="frame-display" id="current-frame-{video_id}">{int(start_time * fps)}</div>
                
                <div class="status">
                    <p><strong>Start Time:</strong> {start_time:.1f}s</p>
                    <p><strong>Start Frame:</strong> {int(start_time * fps)}</p>
                    <p><strong>FPS:</strong> {fps}</p>
                    <p><strong>Status:</strong> <span id="status-{video_id}">Initializing...</span></p>
                    <p><strong>Player State:</strong> <span id="player-state-{video_id}">Unknown</span></p>
                    <div style="margin-top:10px;">
                        <label for="goto-frame-{video_id}"><strong>Go to Frame:</strong></label>
                        <input type="number" id="goto-frame-{video_id}" placeholder="Enter frame and press Enter" style="width: 100%; padding: 6px; border-radius: 4px; border: 1px solid #555; background:#2a2a2a; color:#fff;" />
                    </div>
                    <div style="margin-top:10px;">
                        <div style="display:flex; justify-content:space-between;">
                            <span id="frame-left-{video_id}"><strong>{int(start_time * fps)} F</strong></span>
                            <span id="frame-right-{video_id}"><strong>-- F</strong></span>
                        </div>
                        <input type="range" id="seek-frame-{video_id}" min="0" max="{int(max(1, int(start_time * fps) + 1))}" step="1" value="{int(start_time * fps)}" style="width:100%;">
                    </div>
                </div>
                
                <div class="info-grid">
                    <div class="info-item">
                        <strong>Time Format:</strong><br>
                        <span id="time-formatted-{video_id}">00:13:53</span>
                    </div>
                    <div class="info-item">
                        <strong>Frame Rate:</strong><br>
                        <span id="fps-display-{video_id}">{fps} fps</span>
                    </div>
                </div>
                
                
            </div>
            
            <div class="video-panel">
                <h4>Video Player:</h4>
                <iframe id="ytplayer-{video_id}" 
                        src="https://www.youtube.com/embed/{video_id}?start={start_time}&enablejsapi=1"
                        frameborder="0" 
                        allowfullscreen>
                </iframe>
            </div>
        </div>

        <script>
            var player_{video_id.replace('-', '_')};
            var timeUpdateInterval_{video_id.replace('-', '_')};
            var debugCount_{video_id.replace('-', '_')} = 0;
            var fps = {fps};
            var startTime = {start_time};
            
            // Load YouTube API
            var tag = document.createElement('script');
            tag.src = "https://www.youtube.com/iframe_api";
            var firstScriptTag = document.getElementsByTagName('script')[0];
            firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
            
            function addDebugLog_{video_id.replace('-', '_')}(message) {{
                debugCount_{video_id.replace('-', '_')}++;
                var debugEl = document.getElementById('debug-log-{video_id}');
                var timestamp = new Date().toLocaleTimeString();
                if (debugEl) {{
                    debugEl.innerHTML += `<div>[${{timestamp}}] ${{debugCount_{video_id.replace('-', '_')}}}: ${{message}}</div>`;
                }}
                console.log(`[${{timestamp}}] ${{message}}`);
            }}
            
            function formatTime(seconds) {{
                var hours = Math.floor(seconds / 3600);
                var minutes = Math.floor((seconds % 3600) / 60);
                var secs = Math.floor(seconds % 60);
                return `${{hours.toString().padStart(2, '0')}}:${{minutes.toString().padStart(2, '0')}}:${{secs.toString().padStart(2, '0')}}`;
            }}
            
            function calculateFrame(timeInSeconds) {{
                return Math.floor(timeInSeconds * fps);
            }}
            
            function updateTimeAndFrame_{video_id.replace('-', '_')}() {{
                if (player_{video_id.replace('-', '_')} && typeof player_{video_id.replace('-', '_')}.getCurrentTime === 'function') {{
                    try {{
                        var currentTime = player_{video_id.replace('-', '_')}.getCurrentTime();
                        var currentFrame = calculateFrame(currentTime);
                        var formattedTime = formatTime(currentTime);
                        
                        addDebugLog_{video_id.replace('-', '_')}(`Time: ${{currentTime.toFixed(2)}}s, Frame: ${{currentFrame}}`);
                        
                        // Update time display
                        var timeEl = document.getElementById('current-time-{video_id}');
                        if (timeEl && currentTime !== undefined) {{
                            timeEl.textContent = currentTime.toFixed(1) + 's';
                        }}
                        
                        // Update frame display
                        var frameEl = document.getElementById('current-frame-{video_id}');
                        if (frameEl) {{
                            frameEl.textContent = currentFrame;
                        }}
                        
                        // Update formatted time
                        var formattedEl = document.getElementById('time-formatted-{video_id}');
                        if (formattedEl) {{
                            formattedEl.textContent = formattedTime;
                        }}
                        // Sync frame slider while not dragging
                        var frameSlider = document.getElementById('seek-frame-{video_id}');
                        var frameLeft = document.getElementById('frame-left-{video_id}');
                        if (frameSlider && !window._seeking_{video_id.replace('-', '_')}) {{
                            frameSlider.value = currentFrame;
                            if (frameLeft) frameLeft.innerHTML = '<strong>' + currentFrame + ' F</strong>';
                        }}
                        
                    }} catch (e) {{
                        addDebugLog_{video_id.replace('-', '_')}(`Error getting current time: ${{e.message}}`);
                    }}
                }} else {{
                    addDebugLog_{video_id.replace('-', '_')}('Player not ready or getCurrentTime not available');
                }}
            }}

            // Handle Go to Frame input (seek on Enter)
            document.addEventListener('DOMContentLoaded', function() {{
                var input = document.getElementById('goto-frame-{video_id}');
                if (input) {{
                    input.addEventListener('keydown', function(e) {{
                        if (e.key === 'Enter') {{
                            var f = parseInt(input.value, 10);
                            if (!isNaN(f) && f >= 0) {{
                                var t = f / fps;
                                addDebugLog_{video_id.replace('-', '_')}('Seeking to frame ' + f + ' (t=' + t.toFixed(2) + 's)');
                                if (player_{video_id.replace('-', '_')} && typeof player_{video_id.replace('-', '_')}.seekTo === 'function') {{
                                    try {{
                                        player_{video_id.replace('-', '_')}.seekTo(t, true);
                                    }} catch (err) {{
                                        addDebugLog_{video_id.replace('-', '_')}('Seek error: ' + err.message);
                                    }}
                                }} else {{
                                    addDebugLog_{video_id.replace('-', '_')}('Seek unavailable: player not ready');
                                }}
                            }} else {{
                                addDebugLog_{video_id.replace('-', '_')}('Invalid frame input');
                            }}
                        }}
                    }});
                }}
                // Setup frame slider handlers
                window._seeking_{video_id.replace('-', '_')} = false;
                var frameSlider = document.getElementById('seek-frame-{video_id}');
                var frameLeft = document.getElementById('frame-left-{video_id}');
                if (frameSlider) {{
                    frameSlider.addEventListener('input', function() {{
                        window._seeking_{video_id.replace('-', '_')} = true;
                        var f = parseInt(frameSlider.value, 10) || 0;
                        if (frameLeft) frameLeft.innerHTML = '<strong>' + f + ' F</strong>';
                    }});
                    frameSlider.addEventListener('change', function() {{
                        var f = parseInt(frameSlider.value, 10) || 0;
                        var t = f / fps;
                        addDebugLog_{video_id.replace('-', '_')}('Seeking to frame ' + f + ' (t=' + t.toFixed(2) + 's) via slider');
                        if (player_{video_id.replace('-', '_')} && typeof player_{video_id.replace('-', '_')}.seekTo === 'function') {{
                            try {{ player_{video_id.replace('-', '_')}.seekTo(t, true); }} catch(err) {{ addDebugLog_{video_id.replace('-', '_')}('Seek error: ' + err.message); }}
                        }}
                        window._seeking_{video_id.replace('-', '_')} = false;
                    }});
                    frameSlider.addEventListener('mousedown', function() {{ window._seeking_{video_id.replace('-', '_')} = true; }});
                    frameSlider.addEventListener('mouseup', function() {{ window._seeking_{video_id.replace('-', '_')} = false; }});
                }}
            }});
            
            function onYouTubeIframeAPIReady() {{
                addDebugLog_{video_id.replace('-', '_')}('YouTube API ready, initializing player');
                
                player_{video_id.replace('-', '_')} = new YT.Player('ytplayer-{video_id}', {{
                    events: {{
                        'onReady': onPlayerReady_{video_id.replace('-', '_')},
                        'onStateChange': onPlayerStateChange_{video_id.replace('-', '_')}
                    }}
                }});
            }}
            
            function onPlayerReady_{video_id.replace('-', '_')}(event) {{
                addDebugLog_{video_id.replace('-', '_')}('Player ready, starting time updates');
                document.getElementById('status-{video_id}').textContent = 'Player Ready';
                try {{
                    var dur = player_{video_id.replace('-', '_')}.getDuration ? player_{video_id.replace('-', '_')}.getDuration() : null;
                    var frameSlider = document.getElementById('seek-frame-{video_id}');
                    var frameRight = document.getElementById('frame-right-{video_id}');
                    if (frameSlider && dur && isFinite(dur) && dur > 0) {{
                        var maxF = Math.floor(dur * fps);
                        frameSlider.max = maxF;
                        if (frameRight) frameRight.innerHTML = '<strong>' + maxF + ' F</strong>';
                    }}
                }} catch(_e) {{}}
                startTimeUpdates_{video_id.replace('-', '_')}();
            }}
            
            function onPlayerStateChange_{video_id.replace('-', '_')}(event) {{
                var stateNames = {{
                    0: 'Ended',
                    1: 'Playing', 
                    2: 'Paused',
                    3: 'Buffering',
                    5: 'Cued'
                }};
                var stateName = stateNames[event.data] || 'Unknown';
                addDebugLog_{video_id.replace('-', '_')}(`Player state changed to: ${{event.data}} (${{stateName}})`);
                document.getElementById('player-state-{video_id}').textContent = stateName;
                updateTimeAndFrame_{video_id.replace('-', '_')}();
            }}
            
            function startTimeUpdates_{video_id.replace('-', '_')}() {{
                if (timeUpdateInterval_{video_id.replace('-', '_')}) {{
                    clearInterval(timeUpdateInterval_{video_id.replace('-', '_')});
                }}
                
                timeUpdateInterval_{video_id.replace('-', '_')} = setInterval(function() {{
                    updateTimeAndFrame_{video_id.replace('-', '_')}();
                }}, 200); // Update every 200ms for smooth frame updates
            }}
            
            // Fallback initialization
            setTimeout(function() {{
                if (!player_{video_id.replace('-', '_')} && typeof YT !== 'undefined' && YT.Player) {{
                    addDebugLog_{video_id.replace('-', '_')}('Fallback initialization triggered');
                    onYouTubeIframeAPIReady();
                }} else if (!player_{video_id.replace('-', '_')}) {{
                    addDebugLog_{video_id.replace('-', '_')}('Fallback failed - YT API still not ready');
                }}
            }}, 3000);
            
            addDebugLog_{video_id.replace('-', '_')}('Page loaded, waiting for YouTube API...');
        </script>
    </body>
    </html>
    """
    return html_content

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
        help="Enter your Google Gemini API key. You can get one from https://makersuite.google.com/app/apikey"
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
                help="gemini-2.5-pro: High quality, gemini-2.5-flash: Fast speed"
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
    
    # Test API connection
    if test_button:
        if not api_key:
            st.error("❌ Please enter your Gemini API key first!")
        else:
            with st.spinner("Testing Gemini API connection..."):
                try:
                    # Test with a simple prompt
                    test_prompt = "Hello! Please respond with 'API connection successful' if you can read this message."

                    # Import and test Gemini
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-pro')
                    
                    response = model.generate_content(test_prompt)
                    response_text = response.text.strip()

                    if response_text:
                        st.success(f"✅ **Gemini API Test Successful!**")
                        st.info(f"**Response:** {response_text}")
                        st.markdown("🎉 Your Gemini API is working correctly and ready to use!")
                    else:
                        st.warning("⚠️ **API responded but with empty content**")
                        st.markdown("The API connection works but returned no content.")

                except Exception as e:
                    st.error(f"❌ **Gemini API Test Failed!**")
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
            "BM25 Keyword Search",
        ],
        index=0,
    )

# Derive mode flags from selection
rerank_enabled = (search_mode == "CLIP Multi-aspects Search + Rerank")
bm25_enabled = (search_mode == "BM25 Keyword Search")

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


# Dynamic placeholder based on search mode
if bm25_enabled:
    placeholder_text = """Ví dụ BM25 queries:
• cát AND điêu AND khắc (tìm video có cả 3 từ)
• cát OR "điêu khắc" (tìm video có "cát" hoặc cụm "điêu khắc")
• cát NOT biển (tìm video có "cát" nhưng không có "biển")

Nhấn nút "📚 BM25 Query Syntax" để xem chi tiết syntax."""
else:
    placeholder_text = "Nhập miêu tả truy vấn tiếng Việt / English ..."

query_vi = st.text_area("Enter Query:", height=150, placeholder=placeholder_text)

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
    run_button = st.button("🚀 Run Query", use_container_width=True, type="primary")
    if run_button:
        if not query_vi.strip():
            st.error("Please enter a query first!")
        else:
            run = True; st.session_state['run_flag'] = True
            st.info("🔍 **BM25 Keyword Search** - Searching directly by keywords")
elif gemini_available:
    # CLIP mode with Gemini - show Enhance Query first, then Run Query
    if not st.session_state.query_enhanced:
        # Show Enhance Query button (Green, full width)
        enhance_button = st.button("🌱 Enhance Query", use_container_width=True, type="primary")
        if enhance_button:
            if not query_vi.strip():
                st.error("Please enter a query first!")
            else:
                with st.spinner("Enhancing query with Gemini..."):
                    try:
                        enhanced_data = extract_aspects_and_seeds_with_gemini(
                            query_vi, gemini_temperature, gemini_top_p,
                            gemini_model if 'gemini_model' in locals() else "gemini-2.5-pro",
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
            run_button = st.button("🚀 Run Query", use_container_width=True, type="secondary")
        with cols_rr[1]:
            reset_button = st.button("🔄 Reset Query", use_container_width=True)

        if run_button:
            run = True; st.session_state['run_flag'] = True
        if reset_button:
            st.session_state.query_enhanced = False
            st.session_state.enhanced_data = None
            st.session_state['run_flag'] = False
            st.session_state.pop('clip_cache', None)
            st.rerun()
else:
    # CLIP mode without Gemini - show Run Query directly with fallback
    run_button = st.button("🚀 Run Query", use_container_width=True, type="primary")
    if run_button:
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
        pass  # Don't update log here, we'll handle it manually

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
                                    st.image(keyframe_img, caption=f"Keyframe {actual_frame}", use_container_width=True)
                                else:
                                    # Fallback image if keyframe not found
                                    fallback_img = Image.fromarray(np.full((180, 320, 3), 200, dtype=np.uint8))
                                    st.image(fallback_img, caption=f"No keyframe at ~{sec_to_mmss(timestamp)}", use_container_width=True)

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
                        if url:
                            with st.popover("🎥 Watch Video", use_container_width=True):
                                st.markdown(f"**Video:** {vid}")
                                st.markdown(f"**Timestamp:** {sec_to_mmss(timestamp)}")
                                st.markdown(f"**Keyframe:** {actual_frame}")
                                st.markdown(f"**BM25 Score:** {score:.3f}")

                                # Display highlighted full text in video player
                                highlighted_text = result.get("highlighted_full_text", snippet)
                                st.markdown(f"""
                                    <div class="bm25-fulltext">
                                        <strong>Full Text:</strong> {highlighted_text}
                                    </div>
                                """, unsafe_allow_html=True)

                                actual_video_id = vid
                                youtube_video_id = url.split('v=')[1].split('&')[0] if 'v=' in url else url.split('/')[-1]
                                actual_fps = get_fps_from_csv(actual_video_id)
                                # Use the original timestamp from BM25 results
                                start_sec = int(max(0, timestamp))

                                components.html(create_youtube_player_html(youtube_video_id, start_sec, fps=actual_fps), height=700)
                        else:
                            st.markdown("No video URL available")

                        st.divider()
            else:
                st.warning("No BM25 search results found. Try different keywords or check if BM25 databases exist.")

        except Exception as e:
            st.error(f"BM25 search failed: {e}")
            st.info("Make sure BM25 databases exist in the data directory")

        # Don't continue with CLIP search when BM25 is enabled
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
            gemini_model if 'gemini_model' in locals() else "gemini-2.5-pro",
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
    if "b1" in DATASETS: clip_dirs.append(B1["clip_dir"])  # for visibility in logs
    if "b2" in DATASETS: clip_dirs.append(B2["clip_dir"])  # for visibility in logs
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
                    if url:
                        # Watch Video button with popover
                        with st.popover("🎥 Watch Video", use_container_width=True):
                            st.markdown(f"**Video:** {vid}")
                            st.markdown(f"**Time Window:** {t0s} → {t1s}")
                            st.markdown(f"**Total Score:** {r['total']:.3f}")
                            
                            # Use the actual video ID from dataset (not from YouTube URL)
                            start_sec = int(max(0, win['t0']))
                            actual_video_id = vid  # Use the dataset video ID (e.g., K16_V013)
                            youtube_video_id = url.split('v=')[1].split('&')[0] if 'v=' in url else url.split('/')[-1]
                            
                            # Get actual FPS from CSV file using dataset video ID
                            actual_fps = get_fps_from_csv(actual_video_id)
                            
                            # Use the working YouTube API with FPS-based frame calculation
                            components.html(create_youtube_player_html(youtube_video_id, start_sec, fps=actual_fps), height=700)
                    else:
                        st.markdown("No video URL available")
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
                workers=8, phrases_only=False, max_unigrams=16, max_phrases=10
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
                st.dataframe(_tbl[_show_cols].reset_index(drop=True), use_container_width=True)
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
                        with st.popover("🎥 Watch Video", use_container_width=True):
                            st.markdown(f"**Video:** {vid}")
                            st.markdown(f"**Time Window:** {t0s} → {t1s}")
                            st.markdown(f"**Total Score:** {r['total']:.3f}")
                            
                            # Use the actual video ID from dataset (not from YouTube URL)
                            start_sec = int(max(0, win['t0']))
                            actual_video_id = vid  # Use the dataset video ID (e.g., K16_V013)
                            youtube_video_id = url.split('v=')[1].split('&')[0] if 'v=' in url else url.split('/')[-1]
                            
                            # Get actual FPS from CSV file using dataset video ID
                            actual_fps = get_fps_from_csv(actual_video_id)
                            
                            # Use the working YouTube API with FPS-based frame calculation
                            components.html(create_youtube_player_html(youtube_video_id, start_sec, fps=actual_fps), height=700)
                    else:
                        st.markdown("No video URL available")
                    st.divider()

