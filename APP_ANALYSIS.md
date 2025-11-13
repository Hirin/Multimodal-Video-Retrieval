# Comprehensive App.py Analysis & Refactoring Guide

## Overview
**File:** `/home/user/Multimodal-Video-Retrieval/app.py`
**Lines:** 4459 total
**Type:** Streamlit web application
**Purpose:** Multi-modal video retrieval system combining CLIP embeddings and BM25 text search

---

## 1. IMPORTS & DEPENDENCIES

### Standard Library
- `os`, `io`, `re`, `json`, `glob`, `time`, `sqlite3`, `math`, `unicodedata`
- `concurrent.futures` - Parallel processing
- `pathlib.Path` - File path handling
- `collections` - `deque`, `defaultdict`
- `threading`, `shutil`
- `typing` - Type hints (List, Dict, Tuple, Optional)

### External Libraries
- **Web Framework:** `flask` (Flask, request, Response, send_file) - Video streaming server
- **Data Processing:** `numpy`, `pandas` - Array operations and DataFrames
- **Image Processing:** `PIL.Image` - Image loading and manipulation
- **UI Framework:** `streamlit`, `streamlit.components.v1` - Web UI components
- **NLP/Translation:** `deep_translator.GoogleTranslator` - Language translation
- **ML/AI:**
  - CLIP (via `ensure_clip_text_encoder()`) - Text-image embeddings
  - Google Gemini API (via `google.genai`) - Query enhancement
  - kagglehub - Kaggle dataset downloads

---

## 2. GLOBAL CONFIGURATION & CONSTANTS

### Data Paths
```python
DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/shared/HCMC-AI/")
```

### Batch Datasets (B1 & B2)
Each batch has the following structure:
```python
B1 = {
    "clip_dir": f"{DATA_ROOT}b1/clip-features-32",        # CLIP features (npy files)
    "map_csv": f"{DATA_ROOT}b1/map-keyframes",             # Timestamp-to-keyframe mapping
    "kf_root": f"{DATA_ROOT}b1/keyframes",                 # Keyframe images
    "vid_root": f"{DATA_ROOT}b1/video",                    # Video files
    "keep_dir": f"{DATA_ROOT}b1/clip_keep_idxs",           # Index keep lists
}

B2 = { ... }  # Same structure

B1_DB_PATH = find_db_path(DATA_ROOT, "b1")  # BM25 database paths
B2_DB_PATH = find_db_path(DATA_ROOT, "b2")
```

### Search Parameters
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `DATASETS` | {"b1", "b2"} | Active datasets |
| `WINDOW_SEC` | 40.0 | CLIP window size (seconds) |
| `SKIP_SEC` | 0.0 | Skip intro duration |
| `DELTA_SEC` | 20.0 | Rerank window expansion |
| `TOP_VIDEOS` | 20 | Top K videos to return |
| `LIMIT_VIDEOS` | None | Limit videos searched |
| `DEFAULT_TOP_K` | 20 | Default ranking limit |

### AI/LLM Configuration
```python
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MAX_TOKENS = 10000
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
USE_LLM_ASPECTS = bool(GOOGLE_API_KEY)
```

### Video Serving
```python
LOCAL_VIDEO_ROOT = Path("./videos")
LOCAL_VIDEO_URL_BASE = "http://127.0.0.1:8801/video"
```

### Exclusion Lists
```python
EXCLUDE_SER = set()  # Excluded video series
```

### CLIP Model Cache
```python
_CLIP_RT = {"model": None, "tokenizer": None, "device": None, "backend": None}
```

### Session State Initialization
```python
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    # Clear watch mode keys on app start
```

---

## 3. SESSION STATE VARIABLES

### Core Search State
- `run_flag` - Main search execution flag
- `run_flag_clip` - CLIP search flag
- `run_flag_rerank` - Rerank execution flag
- `run_flag_bm25_seed` - Transcript seed search flag
- `run_flag_bm25_kw` - BM25 keyword search flag
- `run_flag_vid` - Video ID search flag

### Query Enhancement
- `query_enhanced` - Query has been enhanced (bool)
- `enhanced_data` - Cached Gemini/fallback results (dict)
- `auto_run_after_enhance` - Auto-run after enhancement (bool)

### Search Results Cache
- `clip_cache` - CLIP search results
- `rerank_cache` - Rerank results
- `seed_cache` - Seed search results
- `video_cache` - Video player state

### Configuration
- `WINDOW_SEC`, `SKIP_SEC`, `DELTA_SEC`, `TOP_VIDEOS` - Runtime overrides
- `W_CLIP`, `W_TEXT`, `W_ANY`, `W_PHR` - Rerank weights
- `pending_setting_inputs` - Pending settings changes

### UI State
- `last_mode` - Last selected search mode
- `debug_mode` - Debug mode toggle
- `clip_model_preloaded` - CLIP model preloaded
- `turn_off_gemini` - Gemini disabled flag
- `dialog_context` - Dialog context flag
- `video_cache` - Cached video data
- `watch_mode_*` - Watch mode selection per result

---

## 4. FUNCTION DEFINITIONS BY CATEGORY

### 4.1 UI/CSS Functions (Lines 37-143)

| Function | Lines | Purpose |
|----------|-------|---------|
| `create_popover_css()` | 37 | CSS styling for popovers |
| `create_loading_overlay(message)` | 63 | Custom loading animation HTML |
| `reset_watch_mode_keys()` | 132 | Reset video watch mode selections |

### 4.2 Video Download & Management (Lines 147-358)

| Function | Lines | Purpose |
|----------|-------|---------|
| `download_video_dialog()` | 147 | Dialog for Kaggle video downloads |
| `download_video_from_kaggle()` | 326 | Actual download implementation |
| `make_video_app(root_dir)` | 358 | Create Flask app for video serving |
| `ensure_server_once()` | 378 | Start Flask server singleton |
| `find_db_path(data_root, batch_name)` | 422 | Find BM25 database path |

### 4.3 Utility Functions (Lines 465-726)

**Math & Normalization:**
| Function | Lines | Purpose |
|----------|-------|---------|
| `l2_normalize_rows(x)` | 465 | L2 normalize feature vectors |
| `sec_to_mmss(t)` | 470 | Convert seconds to MM:SS format |
| `parse_mmss(s, default)` | 479 | Parse MM:SS to seconds |
| `_as_int(x, default)` | 947 | Safe int conversion |
| `_as_float(x, default)` | 963 | Safe float conversion |

**Path Resolution:**
| Function | Lines | Purpose |
|----------|-------|---------|
| `dataset_for_video(video_id)` | 496 | Determine batch (B1/B2) from video ID |
| `keyframe_path(video_id, frame_id)` | 499 | Get keyframe image path |
| `npy_path_for_video(video_id)` | 514 | Get CLIP features path |
| `map_csv_path_for_video(video_id)` | 518 | Get timestamp map CSV path |
| `keep_idxs_path_for_video(video_id)` | 522 | Get keep indices path |
| `safe_imread(path)` | 526 | Safely load image file |

**Video/Keyframe Processing:**
| Function | Lines | Purpose |
|----------|-------|---------|
| `timestamp_to_keyframe_number(vid, timestamp)` | 536 | Map timestamp to keyframe number |
| `find_best_keyframe(vid, target_frame)` | 575 | Find closest keyframe image |
| `list_videos_from_dirs(clip_dirs, ...)` | 693 | List available videos from directories |
| `load_times_for_video(video_id, n_feats)` | 720 | Load timestamps for CLIP features |
| `get_fps_from_csv(video_id)` | 2565 | Extract FPS from metadata CSV |

### 4.4 Query Enhancement & Seed Extraction (Lines 740-1043)

| Function | Lines | Purpose |
|----------|-------|---------|
| `extract_seeds_from_query(query_vi)` | 740 | Extract keywords/phrases via fallback |
| `get_transcript_snippets_with_highlights()` | 826 | Get transcript matches with highlights |
| `highlight_seeds_in_text(text, unigrams, phrases)` | 877 | Highlight seed keywords in text |
| `get_frame_triplet(video_id, mid_time, delta)` | 908 | Get 3-frame context around timestamp |
| `create_fallback_enhanced_data(query_vi)` | 934 | Create fallback aspects/seeds |
| `extract_aspects_and_seeds_with_gemini()` | 1043 | Use Gemini to generate aspects/seeds |
| `fallback_translate_and_create_aspects()` | 2519 | Translation fallback for aspects |

### 4.5 CLIP Model & Text Encoding (Lines 979-1152)

| Function | Lines | Purpose |
|----------|-------|---------|
| `ensure_clip_text_encoder(model_name, pretrained)` | 979 | Load/initialize CLIP text encoder |
| `encode_text(texts)` | 1001 | Encode text queries to vectors |
| `build_query_vecs_for_aspects(aspects)` | 1138 | Build vector representations for aspects |

### 4.6 CLIP Search & Ranking (Lines 1156-1245)

| Function | Lines | Purpose |
|----------|-------|---------|
| `best_window_multi_aspect()` | 1156 | Find best matching window per aspect |
| `score_video_clip_window(video_id, q_per_aspect, ...)` | 1201 | Score single video with CLIP vectors |
| `rank_videos_clip_window(aspects, ...)` | 1215 | Rank all videos by CLIP similarity |

### 4.7 BM25 Text Search (Lines 1248-1650)

**Query Building:**
| Function | Lines | Purpose |
|----------|-------|---------|
| `_tok_vi(s)` | 1248 | Vietnamese text tokenization |
| `build_keyword_sets()` | 1251 | Build unigrams + phrases from query |
| `build_bm25_query_from_seeds()` | 1263 | Convert seeds to BM25 query string |
| `convert_bm25_syntax(query)` | 1381 | Convert user syntax to SQLite FTS5 |
| `_vi_strip_accents(s)` | 1752 | Remove Vietnamese diacritics |
| `_fts5_escape_term(term)` | 1755 | Escape special characters for FTS5 |
| `_build_seed_fts5_query(seeds_json)` | 1769 | Build FTS5 query from seeds |

**Database Operations:**
| Function | Lines | Purpose |
|----------|-------|---------|
| `_connect_ro(db_path)` | 1273 | Open read-only DB connection |
| `_count_one()` | 1276 | Count matches in time window |
| `_alias_windows_and_dbpaths()` | 1288 | Group results by dataset |
| `_run_fts5_query()` | 1791 | Execute FTS5 query on database |
| `_resolve_bm25_dbpaths()` | 1859 | Resolve BM25 database paths |

**Search & Filtering:**
| Function | Lines | Purpose |
|----------|-------|---------|
| `bm25_search_videos(query, limit)` | 1547 | Main BM25 keyword search |
| `load_keyframes_data()` | 1401 | Load keyframe metadata |
| `get_keyframes_data()` | 1464 | Get cached keyframe data |
| `apply_search_filters()` | 1471 | Apply exclude filters to results |
| `deduplicate_bm25_results()` | 1499 | Remove near-duplicate results |
| `transcript_seed_bm25_search()` | 1888 | BM25 search using seeds |

**Text Processing:**
| Function | Lines | Purpose |
|----------|-------|---------|
| `highlight_keywords_in_text(text, query)` | 1297 | Highlight BM25 query terms |
| `_count_unigram_hits()` | 1821 | Count unigram matches in text |
| `_count_phrase_hits()` | 1843 | Count phrase matches in text |
| `_window_key(t, window_sec, stride_sec)` | 1850 | Generate window time key |

### 4.8 Reranking (Lines 1650-1756)

| Function | Lines | Purpose |
|----------|-------|---------|
| `rerank_with_transcript_ultra_window()` | 1650 | Rerank CLIP results with transcript scores |
| `reorder_results_by_df()` | 1745 | Reorder results by DataFrame |

### 4.9 Video Player Creation (Lines 2000-2518)

| Function | Lines | Purpose |
|----------|-------|---------|
| `_create_base_player_html()` | 2000 | Base player HTML template |
| `create_youtube_player_html()` | 2361 | YouTube iframe player with controls |
| `create_local_player_html()` | 2495 | Local video player (HTML5) |

### 4.10 Video Info & URLs (Lines 2565-2827)

| Function | Lines | Purpose |
|----------|-------|---------|
| `get_watch_url(video_id)` | 2587 | Get YouTube URL for video |
| `get_video_info(video_id)` | 2606 | Get video metadata |
| `_mode_key(name)` | 2829 | Map mode name to key |
| `_clear_run_flags()` | 2845 | Clear search execution flags |
| `_get_run_flag(mode_key)` | 2850 | Get flag for mode |
| `_set_run_flag(mode_key, value)` | 2853 | Set flag for mode |

### 4.11 Logging & Utilities (Lines 3305+)

| Function | Lines | Purpose |
|----------|-------|---------|
| `run_with_logs(fn, *args, **kwargs)` | 3305 | Execute function with log capture |

---

## 5. MAIN APPLICATION FLOW

### Entry Point Structure (Lines 2630-3305+)

```
1. Page Config (Line 2630)
   ├─ Load CLIP model (2633)
   └─ Initialize session state

2. Sidebar Configuration (2645)
   ├─ Search mode selection
   ├─ Dataset filters
   ├─ Gemini API setup
   ├─ Search parameters
   └─ Debug mode toggle

3. Main Content Area (2826+)
   ├─ Mode key assignment
   ├─ Query input (based on mode)
   ├─ Button logic (Enhance/Run/Reset)
   └─ Results display (dynamic)

4. Search Execution (3394+)
   ├─ BM25 Keyword Search
   ├─ CLIP Multi-aspect Search
   ├─ Transcript Seed Search
   ├─ Reranking
   ├─ Video ID Search
   └─ Result rendering with video player
```

### Search Mode Logic

| Mode | Flow | Line |
|------|------|------|
| **CLIP Multi-aspects** | Enhance (Gemini) → Run → Results | 3209+ |
| **CLIP + Rerank** | Enhance → Run → CLIP + Rerank | 3256 |
| **Transcript Seed** | Enhance (Gemini or Manual) → Run | 3108+ |
| **BM25 Keyword** | Run Query directly | 3078+ |
| **Video ID Search** | Search by ID directly | 2965+ |

---

## 6. EXTERNAL SERVICES & INTEGRATIONS

### Google Gemini API
**Purpose:** Query enhancement via aspect extraction
**Functions:**
- `extract_aspects_and_seeds_with_gemini()` (Line 1043)
- Fallback to translation-based approach if API unavailable

**Configuration:**
- API Key: `GOOGLE_API_KEY`
- Model: `gemini-2.5-flash` or `gemini-2.5-pro`
- Max Tokens: 10000
- Temperature & Top-P configurable (0-2.0, 0-1.0)

### CLIP (Contrastive Language-Image Pre-training)
**Purpose:** Text-to-feature vector encoding
**Functions:**
- `ensure_clip_text_encoder()` (Line 979)
- `encode_text()` (Line 1001)

**Models:**
- Default: `ViT-B-32` (OpenAI pretrained)
- Features: Pre-computed and stored as `.npy` files

### SQLite FTS5 (BM25)
**Purpose:** Full-text search on video transcripts
**Database Schema:**
```sql
CREATE VIRTUAL TABLE docs USING fts5(
    video_id,      -- Video identifier
    text,          -- Full transcript
    t,             -- Timestamp
    ...
)
```
**Paths:** `B1_DB_PATH`, `B2_DB_PATH`

### Kaggle Dataset API
**Purpose:** Download missing video files
**Datasets:**
- Batch 1: `ndmhoang/hcm25-batch1-videos`
- Batch 2: `duyminhhong/hcmai25-batch2-videos`

**Configuration:**
- Requires `.kaggle/kaggle.json` in working directory
- Used by `download_video_dialog()` (Line 147)

### YouTube API
**Purpose:** Video streaming via embedded iframe
**Functions:**
- `create_youtube_player_html()` (Line 2361)
- Includes seeking, frame seeking, time display

### Flask Server
**Purpose:** Serve local video files
**Port:** 8801
**Functions:**
- `make_video_app()` (Line 358)
- `ensure_server_once()` (Line 378)

---

## 7. LOGICAL MODULE BREAKDOWN

### **config/** - Configuration & Settings
```
├── default_config.py (Lines 400-462)
│   ├── DATA_ROOT, DATASETS
│   ├── B1, B2 path definitions
│   ├── WINDOW_SEC, SKIP_SEC, DELTA_SEC, TOP_VIDEOS
│   ├── Gemini parameters
│   └── Local video serving config
│
├── gemini_config.py (Lines 450-454)
│   ├── GEMINI_MODEL, GEMINI_MAX_TOKENS
│   ├── USE_LLM_ASPECTS
│   └── API key management
│
├── db_config.py (Lines 422-447)
│   └── find_db_path(), B1_DB_PATH, B2_DB_PATH
│
└── session_state.py (Lines 28-34, 2630-2827)
    ├── Session initialization
    ├── State variable initialization
    └── Configuration UI (sidebar)
```

### **models/** - Data Models & Structures
```
├── video_model.py
│   ├── Video metadata structure
│   └── Batch/Dataset definitions
│
├── search_query_model.py
│   ├── Query aspects
│   ├── Seeds (unigrams, phrases)
│   └── Search results structure
│
├── clip_features_model.py
│   ├── Feature vectors
│   ├── Timestamps
│   └── Keyframe mappings
│
└── bm25_result_model.py
    ├── BM25 search results
    └── Highlighted text
```

### **services/** - Business Logic
```
├── search_service/
│   ├── clip_search.py (Lines 1156-1245)
│   │   ├── rank_videos_clip_window()
│   │   ├── score_video_clip_window()
│   │   └── best_window_multi_aspect()
│   │
│   ├── bm25_search.py (Lines 1547-1650)
│   │   ├── bm25_search_videos()
│   │   ├── transcript_seed_bm25_search()
│   │   ├── convert_bm25_syntax()
│   │   └── build_keyword_sets()
│   │
│   └── rerank_service.py (Lines 1650-1756)
│       ├── rerank_with_transcript_ultra_window()
│       └── reorder_results_by_df()
│
├── query_service/
│   ├── query_enhancer.py (Lines 1043-1135)
│   │   ├── extract_aspects_and_seeds_with_gemini()
│   │   └── fallback_translate_and_create_aspects()
│   │
│   ├── seed_extractor.py (Lines 740-908)
│   │   ├── extract_seeds_from_query()
│   │   ├── get_transcript_snippets_with_highlights()
│   │   └── highlight_seeds_in_text()
│   │
│   └── clip_encoder.py (Lines 979-1152)
│       ├── ensure_clip_text_encoder()
│       ├── encode_text()
│       └── build_query_vecs_for_aspects()
│
├── video_service/
│   ├── video_loader.py (Lines 536-720)
│   │   ├── find_best_keyframe()
│   │   ├── timestamp_to_keyframe_number()
│   │   ├── load_times_for_video()
│   │   └── list_videos_from_dirs()
│   │
│   ├── video_downloader.py (Lines 147-358)
│   │   ├── download_video_dialog()
│   │   ├── download_video_from_kaggle()
│   │   └── reset_download_state()
│   │
│   ├── video_metadata.py (Lines 2565-2606)
│   │   ├── get_watch_url()
│   │   ├── get_video_info()
│   │   └── get_fps_from_csv()
│   │
│   └── video_server.py (Lines 358-420)
│       ├── make_video_app()
│       └── ensure_server_once()
│
└── database_service/
    ├── bm25_db.py (Lines 1273-1791)
    │   ├── _connect_ro()
    │   ├── _run_fts5_query()
    │   ├── _count_one()
    │   └── _resolve_bm25_dbpaths()
    │
    └── keyframe_db.py (Lines 1401-1499)
        ├── load_keyframes_data()
        ├── get_keyframes_data()
        └── apply_search_filters()
```

### **utils/** - Utility Functions
```
├── math_utils.py (Lines 465-495)
│   ├── l2_normalize_rows()
│   ├── sec_to_mmss()
│   └── parse_mmss()
│
├── path_utils.py (Lines 496-535)
│   ├── dataset_for_video()
│   ├── keyframe_path()
│   ├── npy_path_for_video()
│   ├── map_csv_path_for_video()
│   ├── keep_idxs_path_for_video()
│   └── safe_imread()
│
├── text_utils.py (Lines 1248-1379)
│   ├── _tok_vi()
│   ├── _vi_strip_accents()
│   ├── _fts5_escape_term()
│   ├── highlight_keywords_in_text()
│   ├── _count_unigram_hits()
│   ├── _count_phrase_hits()
│   └── _window_key()
│
├── type_utils.py (Lines 947-978)
│   ├── _as_int()
│   └── _as_float()
│
└── logging_utils.py (Lines 3305+)
    └── run_with_logs()
```

### **components/** - UI Components
```
├── css_components.py (Lines 37-59)
│   ├── create_popover_css()
│   └── UI styling utilities
│
├── loading_components.py (Lines 63-130)
│   ├── create_loading_overlay()
│   └── Progress/loading animations
│
├── sidebar_components.py (Lines 2645-2827)
│   ├── Search mode selector
│   ├── Dataset filters
│   ├── Gemini configuration UI
│   ├── Search parameters UI
│   └── Debug mode toggle
│
├── search_input_components.py (Lines 2873-2991)
│   ├── Max aspects input
│   ├── Query text area
│   ├── Video ID input
│   ├── BM25 syntax help
│   └── Button styling
│
├── results_components.py (Lines 3394-3700)
│   ├── BM25 results display
│   ├── CLIP results display
│   ├── Rerank results display
│   ├── Video player integration
│   └── Metadata display
│
├── player_components.py (Lines 2000-2518)
│   ├── _create_base_player_html()
│   ├── create_youtube_player_html()
│   └── create_local_player_html()
│
├── download_dialog.py (Lines 147-325)
│   └── download_video_dialog()
│
└── logs_components.py (Lines 3301-3305)
    └── Log expander and container
```

---

## 8. DATA FLOW DIAGRAMS

### CLIP Multi-aspect Search Flow
```
User Query (Vietnamese/English)
    ↓
[Enhance Query] → extract_aspects_and_seeds_with_gemini()
    ↓
[Aspects] → build_query_vecs_for_aspects() → encode_text() → CLIP vectors
    ↓
[Run Query] → rank_videos_clip_window()
    ├─ For each video:
    │  ├─ Load CLIP features (npy)
    │  ├─ score_video_clip_window()
    │  └─ best_window_multi_aspect()
    ↓
Sort by total score
    ↓
Display results with keyframes & timestamps
```

### BM25 Keyword Search Flow
```
User Query (BM25 syntax)
    ↓
convert_bm25_syntax() → SQLite FTS5 syntax
    ↓
[Run Query] → bm25_search_videos()
    ├─ Query B1_DB_PATH
    ├─ Query B2_DB_PATH
    ↓
highlight_keywords_in_text() → highlighted results
    ↓
apply_search_filters() → apply exclusions
    ↓
deduplicate_bm25_results() → remove near-duplicates
    ↓
Display results with highlighted text
```

### Transcript Seed Search Flow
```
User Query (Vietnamese/English)
    ↓
[Enhance Query] → extract_aspects_and_seeds_with_gemini() or create_fallback_enhanced_data()
    ├─ Extract: aspects, unigrams, phrases
    ↓
[Run Query] → transcript_seed_bm25_search()
    ├─ rank_videos_clip_window() → CLIP results
    ├─ For each CLIP result:
    │  └─ rerank_with_transcript_ultra_window()
    │     ├─ build_keyword_sets() from seeds
    │     ├─ Query BM25 databases
    │     ├─ Calculate text scores (unigram + phrase hits)
    │     └─ Fuse: w_clip * clip_score + w_text * text_score
    ↓
Display reranked results
```

---

## 9. KEY ALGORITHMIC COMPONENTS

### Best Window Multi-aspect (Lines 1156-1199)
**Algorithm:** Sliding window with deque-based maximum finding
- Maintains per-aspect similarity scores
- Uses deques to efficiently find max scores in sliding windows
- Returns: Best window for each aspect + combined score

### CLIP Window Scoring (Lines 1201-1213)
**Process:**
1. Load video's pre-computed CLIP features (npy file)
2. L2-normalize features
3. For each aspect query vector: compute similarity with all frames
4. Find best window using `best_window_multi_aspect()`

### BM25 Full-text Search (Lines 1547-1648)
**Process:**
1. Convert user-friendly syntax to SQLite FTS5
2. Execute MATCH query on FTS5 virtual table
3. Use `snippet()` for context extraction
4. Highlight matching terms with regex

### Reranking with Transcript (Lines 1650-1744)
**Scoring Formula:**
```
final_score = w_clip * (clip_total / num_aspects)
            + w_text * (w_any * unigram_hits + w_phr * phrase_hits)
```
**Process:**
1. Get top-N CLIP results
2. For each result's window, count transcript matches
3. Fuse CLIP + text scores
4. Reorder by combined score

---

## 10. DATASET STRUCTURE

### File Organization
```
$DATA_ROOT/
├── b1/
│   ├── clip-features-32/          # CLIP features (L{XY}_V{ZZZ}.npy)
│   ├── clip_keep_idxs/            # Feature index lists
│   ├── map-keyframes/             # Timestamp CSV files
│   ├── keyframes/                 # Keyframe images (L{XY}/V{ZZZ}/)
│   ├── video/                     # Original videos
│   └── text_bm25.db               # BM25 database
│
└── b2/
    ├── clip-features-32/
    ├── clip_keep_idxs/
    ├── map-keyframes/
    ├── keyframes/
    ├── video/
    └── text_bm25.db
```

### Video ID Format
- Batch 1: `L{XX}_V{ZZZ}` (e.g., `L01_V001`)
- Batch 2: `K{XX}_V{ZZZ}` (e.g., `K01_V001`)

### Feature Files
- `.npy` files: numpy arrays of CLIP features
- `.csv` files: timestamp-to-keyframe mappings
- Image formats: `.jpg`, `.jpeg`, `.png`

---

## 11. REFACTORING PRIORITIES

### High Priority (Core Logic)
1. **Extract search services** (CLIP, BM25, Rerank) → `services/search_service/`
2. **Separate query enhancement** (Gemini, fallback) → `services/query_service/`
3. **Extract video utilities** (loading, metadata) → `services/video_service/`
4. **Separate database operations** (FTS5, queries) → `services/database_service/`

### Medium Priority (Configuration & Utilities)
5. **Extract configuration** → `config/` module
6. **Separate utility functions** → `utils/` module
7. **Extract UI components** → `components/` module

### Low Priority (Code Quality)
8. Remove code duplication
9. Add comprehensive error handling
10. Add logging and monitoring

---

## 12. CRITICAL DEPENDENCIES

### External APIs
- Google Gemini API (gemini-2.5-flash)
- YouTube Embed API
- Kaggle API (optional, for downloads)

### Local Files
- CLIP features (npy files) - **Required** for CLIP search
- BM25 databases (SQLite) - **Required** for text search
- Keyframe images - **Required** for UI display
- Metadata CSVs - **Required** for timestamp mapping

### Python Packages
- streamlit >= 1.0
- numpy >= 1.20
- pandas >= 1.0
- flask >= 2.0
- pillow >= 8.0
- transformers (for CLIP)
- google-genai (for Gemini)
- deep_translator (for Google Translate fallback)

---

## 13. TESTING CONSIDERATIONS

### Unit Test Categories
1. **Path utilities** - Video ID parsing, path generation
2. **Text processing** - Vietnamese tokenization, diacritics removal
3. **Math functions** - L2 normalization, time conversion
4. **Search algorithms** - Window scoring, ranking

### Integration Test Categories
1. **CLIP search end-to-end**
2. **BM25 search with filtering**
3. **Reranking logic**
4. **Video player generation**
5. **Gemini API integration**

### Mock Requirements
- CLIP model (expensive, preload once)
- Gemini API (use test responses)
- Databases (use small test databases)
- Keyframe images (use placeholder images)

---

## 14. CONFIGURATION PARAMETERS SUMMARY

| Parameter | Type | Default | Mutable | Impact |
|-----------|------|---------|---------|--------|
| `WINDOW_SEC` | float | 40.0 | Yes | CLIP window size |
| `SKIP_SEC` | float | 0.0 | Yes | Skip intro duration |
| `DELTA_SEC` | float | 20.0 | Yes | Rerank window expansion |
| `TOP_VIDEOS` | int | 20 | Yes | Results limit |
| `DATASETS` | set | {"b1","b2"} | Yes | Active datasets |
| `W_CLIP` | float | 0.4 | Yes | Rerank CLIP weight |
| `W_TEXT` | float | 0.6 | Yes | Rerank text weight |
| `W_ANY` | float | 0.3 | Yes | Unigram weight |
| `W_PHR` | float | 0.7 | Yes | Phrase weight |
| `GEMINI_MODEL` | str | "gemini-2.5-flash" | No | LLM model |
| `GOOGLE_API_KEY` | str | (env) | Yes | Gemini auth |

---

## 15. PERFORMANCE OPTIMIZATION OPPORTUNITIES

1. **Caching**
   - Cache CLIP model in memory
   - Cache encoded query vectors
   - Cache database queries

2. **Parallelization**
   - Parallel video scoring (already using concurrent.futures)
   - Parallel database queries

3. **Indexing**
   - Pre-compute normalized features
   - Index keyframe locations
   - FTS5 already indexed

4. **Memory Management**
   - Lazy load videos
   - Stream large features
   - Use generators for large result sets

5. **Database Optimization**
   - Connection pooling
   - Query caching
   - Index analysis

---

This comprehensive breakdown provides the foundation for systematically refactoring app.py into a modular, maintainable structure while preserving all functionality.
