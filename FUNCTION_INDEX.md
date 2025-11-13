# Function Index & Quick Reference

## All Functions by Category

### CLIP Search & Ranking (6 functions)

| Function | Lines | LOC | Purpose | Inputs | Output |
|----------|-------|-----|---------|--------|--------|
| `encode_text()` | 1001 | 42 | Encode text prompts to CLIP vectors | `texts: List[str]` | `np.ndarray (N, 512)` |
| `build_query_vecs_for_aspects()` | 1138 | 14 | Organize aspect prompts into vectors | `aspects: List[Dict]` | `(vecs, names, weights)` |
| `best_window_multi_aspect()` | 1156 | 44 | Find best time window for multi-aspect | `feats, times, q_per_aspect, weights` | `Dict with window info` |
| `score_video_clip_window()` | 1201 | 12 | Score single video with CLIP | `video_id, q_per_aspect, weights` | `Dict with score` |
| `rank_videos_clip_window()` | 1215 | 29 | Rank all videos by CLIP similarity | `aspects, limit_videos, ...filters` | `List[Dict] ranked results` |
| `ensure_clip_text_encoder()` | 979 | 21 | Load/cache CLIP text encoder | `model_name, pretrained` | `None (loads into global)` |

---

### BM25 Search (12 functions)

| Function | Lines | LOC | Purpose | Inputs | Output |
|----------|-------|-----|---------|--------|--------|
| `bm25_search_videos()` | 1547 | 102 | Main BM25 keyword search | `query: str, limit: int` | `List[Dict] with results` |
| `convert_bm25_syntax()` | 1381 | 19 | User syntax → FTS5 syntax | `query: str` | `str (converted query)` |
| `build_keyword_sets()` | 1251 | 12 | Extract unigrams + phrases | `query_vi, seeds, max_uni, max_phr` | `(uni, phr, q_any, q_phr)` |
| `build_bm25_query_from_seeds()` | 1263 | 9 | Combine seeds into MATCH query | `query_vi, seeds` | `str (query)` |
| `highlight_keywords_in_text()` | 1297 | 85 | Highlight matching keywords | `text: str, query: str` | `str (HTML with <b> tags)` |
| `apply_search_filters()` | 1471 | 28 | Apply exclusion filters | `results, exclude_series, exclude_vids` | `List[Dict] (filtered)` |
| `deduplicate_bm25_results()` | 1499 | 48 | Remove near-duplicate results | `results, time_threshold` | `List[Dict] (deduplicated)` |
| `_run_fts5_query()` | 1791 | 29 | Execute FTS5 query on DB | `db_path, match_query, limit` | `List[tuples] (rows)` |
| `_count_one()` | 1276 | 11 | Count matches in time window | `db_path, vid, t0, t1, q_any, q_phr` | `(vid, h_any, h_phr)` |
| `transcript_seed_bm25_search()` | 1888 | 112 | BM25 search using seeds | `query_vi, seeds, ...params` | `List[Dict] (ranked results)` |
| `_build_seed_fts5_query()` | 1769 | 22 | Build FTS5 query from seeds | `seeds_json: dict` | `str (query)` |
| `_alias_windows_and_dbpaths()` | 1288 | 8 | Group results by batch | `windows` | `List[(batch, path, windows)]` |

---

### Text Processing & Utilities (10 functions)

| Function | Lines | LOC | Purpose | Inputs | Output |
|----------|-------|-----|---------|--------|--------|
| `_tok_vi()` | 1248 | 1 | Tokenize Vietnamese text | `s: str` | `List[str] (tokens)` |
| `_vi_strip_accents()` | 1752 | 2 | Remove Vietnamese diacritics | `s: str` | `str` |
| `_fts5_escape_term()` | 1755 | 13 | Escape FTS5 special chars | `term: str` | `str` |
| `_count_unigram_hits()` | 1821 | 22 | Count unigram matches | `text_low, unigrams_low` | `int (count)` |
| `_count_phrase_hits()` | 1843 | 7 | Count phrase matches | `text_low, phrases_low` | `int (count)` |
| `_window_key()` | 1850 | 9 | Generate window time key | `t, window_sec, stride_sec` | `float (key)` |
| `highlight_seeds_in_text()` | 877 | 31 | Highlight seeds in snippet | `text, unigrams, phrases` | `str` |
| `get_transcript_snippets_with_highlights()` | 826 | 51 | Get transcript matches | `vid, t0, t1, seeds, top_n` | `List[Dict] (snippets)` |
| `_connect_ro()` | 1273 | 2 | Open read-only DB | `db_path: str` | `sqlite3.Connection` |
| `_resolve_bm25_dbpaths()` | 1859 | 28 | Resolve DB paths | `db_batch1, db_batch2` | `(b1_path, b2_path)` |

---

### Query Enhancement (3 functions)

| Function | Lines | LOC | Purpose | Inputs | Output |
|----------|-------|-----|---------|--------|--------|
| `extract_aspects_and_seeds_with_gemini()` | 1043 | 95 | Use Gemini to enhance query | `query_vi, temp, top_p, model` | `Dict (aspects, seeds)` |
| `fallback_translate_and_create_aspects()` | 2519 | 46 | Translation-based fallback | `query_vi` | `(aspects, seeds)` |
| `create_fallback_enhanced_data()` | 934 | 13 | Create fallback data | `query_vi` | `Dict (enhanced_data)` |
| `extract_seeds_from_query()` | 740 | 87 | Extract seeds (original) | `query_vi: str` | `Dict (seeds)` |

---

### Video Loading & Processing (9 functions)

| Function | Lines | LOC | Purpose | Inputs | Output |
|----------|-------|-----|---------|--------|--------|
| `find_best_keyframe()` | 575 | 118 | Find closest keyframe image | `vid, target_frame` | `(Image, actual_frame)` |
| `timestamp_to_keyframe_number()` | 536 | 39 | Map timestamp to frame # | `vid, timestamp` | `int (frame)` |
| `load_times_for_video()` | 720 | 20 | Load timestamps for video | `video_id, n_feats` | `np.ndarray (times)` |
| `list_videos_from_dirs()` | 693 | 27 | List available videos | `clip_dirs, limit, filters` | `List[str] (video_ids)` |
| `safe_imread()` | 526 | 9 | Safely load image | `path: str` | `Image or None` |
| `get_frame_triplet()` | 908 | 26 | Get 3-frame context | `vid, mid_time, delta` | `List[Image]` |
| `keyframe_path()` | 499 | 13 | Get keyframe file path | `vid, frame_id` | `str (path)` |
| `get_fps_from_csv()` | 2565 | 21 | Extract FPS from metadata | `video_id` | `float (fps)` |
| `get_video_info()` | 2606 | 223 | Get full video metadata | `video_id` | `Dict or None` |

---

### Path & Dataset Utilities (6 functions)

| Function | Lines | LOC | Purpose | Inputs | Output |
|----------|-------|-----|---------|--------|--------|
| `dataset_for_video()` | 496 | 1 | Determine batch from ID | `video_id: str` | `str ("b1" or "b2")` |
| `npy_path_for_video()` | 514 | 2 | Get CLIP features path | `video_id: str` | `str (path)` |
| `map_csv_path_for_video()` | 518 | 2 | Get timestamp map path | `video_id: str` | `str (path)` |
| `keep_idxs_path_for_video()` | 522 | 2 | Get keep indices path | `video_id: str` | `str (path)` |
| `find_db_path()` | 422 | 23 | Find BM25 database | `data_root, batch_name` | `str (path) or None` |
| `get_watch_url()` | 2587 | 18 | Get YouTube URL | `video_id` | `str (url) or None` |

---

### Math & Conversion Utilities (5 functions)

| Function | Lines | LOC | Purpose | Inputs | Output |
|----------|-------|-----|---------|--------|--------|
| `l2_normalize_rows()` | 465 | 4 | L2 normalize vectors | `x: np.ndarray` | `np.ndarray` |
| `sec_to_mmss()` | 470 | 8 | Seconds → MM:SS | `t: float` | `str` |
| `parse_mmss()` | 479 | 16 | MM:SS → seconds | `s: str, default` | `float` |
| `_as_int()` | 947 | 16 | Safe int conversion | `x, default` | `int` |
| `_as_float()` | 963 | 16 | Safe float conversion | `x, default` | `float` |

---

### Video Download & Server (5 functions)

| Function | Lines | LOC | Purpose | Inputs | Output |
|----------|-------|-----|---------|--------|--------|
| `download_video_dialog()` | 147 | 179 | Show download dialog | `video_id, local_root` | `Path or None` |
| `download_video_from_kaggle()` | 326 | 32 | Download from Kaggle | `video_id, local_root` | `Path or None` |
| `make_video_app()` | 358 | 20 | Create Flask app | `root_dir: str` | `Flask app` |
| `ensure_server_once()` | 378 | 44 | Start Flask server | `app, host, port` | `None (starts thread)` |
| `reset_watch_mode_keys()` | 132 | 11 | Reset UI state | `None` | `None` |

---

### Video Player Creation (3 functions)

| Function | Lines | LOC | Purpose | Inputs | Output |
|----------|-------|-----|---------|--------|--------|
| `_create_base_player_html()` | 2000 | 361 | Base player HTML | `video_id, start_time, fps, ...` | `str (HTML)` |
| `create_youtube_player_html()` | 2361 | 134 | YouTube iframe player | `youtube_id, start_time, fps, ...` | `str (HTML)` |
| `create_local_player_html()` | 2495 | 24 | HTML5 video player | `file_url, start_time, fps, ...` | `str (HTML)` |

---

### Reranking (2 functions)

| Function | Lines | LOC | Purpose | Inputs | Output |
|----------|-------|-----|---------|--------|--------|
| `rerank_with_transcript_ultra_window()` | 1650 | 95 | Rerank CLIP with text | `clip_results, query_vi, seeds, ...` | `pd.DataFrame` |
| `reorder_results_by_df()` | 1745 | 6 | Reorder by DataFrame | `orig_results, ranked_df` | `List[Dict]` |

---

### Keyframe Data Management (3 functions)

| Function | Lines | LOC | Purpose | Inputs | Output |
|----------|-------|-----|---------|--------|--------|
| `load_keyframes_data()` | 1401 | 63 | Load keyframe metadata | `None` | `Dict` |
| `get_keyframes_data()` | 1464 | 6 | Get cached data | `None` | `Dict` |
| `_keyframes_cache` | 1462 | - | Global cache | - | - |

---

### UI/CSS (3 functions)

| Function | Lines | LOC | Purpose | Inputs | Output |
|----------|-------|-----|---------|--------|--------|
| `create_popover_css()` | 37 | 22 | Popover styling | `None` | `str (CSS)` |
| `create_loading_overlay()` | 63 | 68 | Loading animation | `message: str` | `str (HTML)` |
| `run_with_logs()` | 3305 | 17 | Execute with logging | `fn, *args, **kwargs` | `(result, StringIO)` |

---

### Session State Helpers (3 functions)

| Function | Lines | LOC | Purpose | Inputs | Output |
|----------|-------|-----|---------|--------|--------|
| `_mode_key()` | 2829 | 7 | Map mode → key | `name: str` | `str` |
| `_clear_run_flags()` | 2845 | 4 | Clear search flags | `except_mode_key` | `None` |
| `_get_run_flag()` | 2850 | 1 | Get flag value | `mode_key: str` | `bool` |
| `_set_run_flag()` | 2853 | 1 | Set flag value | `mode_key, value` | `None` |

---

## Function Call Graph (Dependencies)

### Main Search Flows

#### CLIP Search Flow
```
rank_videos_clip_window()
├─ list_videos_from_dirs()
├─ build_query_vecs_for_aspects()
│  ├─ encode_text()
│  │  └─ ensure_clip_text_encoder()
│  └─ l2_normalize_rows()
├─ score_video_clip_window() (per video)
│  ├─ npy_path_for_video()
│  ├─ load_times_for_video()
│  └─ best_window_multi_aspect()
│     └─ l2_normalize_rows()
└─ [Return sorted results]
```

#### BM25 Search Flow
```
bm25_search_videos()
├─ convert_bm25_syntax()
├─ For each database (B1, B2):
│  ├─ _connect_ro()
│  └─ Execute MATCH query
│     ├─ highlight_keywords_in_text()
│     └─ highlight_seeds_in_text()
├─ apply_search_filters()
└─ deduplicate_bm25_results()
```

#### Transcript Seed Search Flow
```
transcript_seed_bm25_search()
├─ extract_aspects_and_seeds_with_gemini()
│  └─ fallback_translate_and_create_aspects()
├─ rank_videos_clip_window()
└─ rerank_with_transcript_ultra_window()
   ├─ build_keyword_sets()
   ├─ _run_fts5_query()
   │  └─ _build_seed_fts5_query()
   └─ reorder_results_by_df()
```

#### Result Display Flow
```
Display Result
├─ timestamp_to_keyframe_number()
├─ find_best_keyframe()
│  ├─ keyframe_path()
│  └─ safe_imread()
├─ get_watch_url()
├─ create_youtube_player_html() OR create_local_player_html()
│  └─ _create_base_player_html()
├─ get_transcript_snippets_with_highlights()
└─ Display with metadata
   └─ sec_to_mmss()
```

---

## Module Dependency Matrix

```
                                Depends On
Function                        utils  config  models  services
─────────────────────────────────────────────────────────────────
Search Module:
  rank_videos_clip_window()      •        •              •
  bm25_search_videos()           •        •              •
  transcript_seed_search()       •        •              •

Query Module:
  extract_aspects_and_gemini()   •        •              •
  encode_text()                  •        •

Video Module:
  find_best_keyframe()           •        •
  load_times_for_video()         •        •
  get_video_info()               •        •

Database Module:
  _run_fts5_query()              •        •
  _connect_ro()                  •

UI Module:
  create_youtube_player_html()                          •
  create_loading_overlay()       •
  render_sidebar()                        •        •

Utils Module:
  l2_normalize_rows()
  sec_to_mmss()
  dataset_for_video()
```

---

## Critical Paths (Performance)

### Most Called Functions (in search execution)
1. `l2_normalize_rows()` - Called for every video's features
2. `best_window_multi_aspect()` - Called for every video
3. `_run_fts5_query()` - Called for every BM25 search
4. `highlight_keywords_in_text()` - Called for every result

### Most I/O Heavy
1. `find_best_keyframe()` - Image loading
2. `load_times_for_video()` - CSV reading
3. `_run_fts5_query()` - Database query
4. `get_video_info()` - Full metadata CSV

### Most Memory Heavy
1. `rank_videos_clip_window()` - Loads all video features into memory
2. `encode_text()` - CLIP model inference
3. `rerank_with_transcript_ultra_window()` - Creates large DataFrames

---

## Testing Priorities

### Unit Tests (by importance)
1. **Math functions** - `l2_normalize_rows()`, `sec_to_mmss()`, `parse_mmss()`
2. **Path functions** - `dataset_for_video()`, `keyframe_path()`, etc.
3. **Text processing** - `_tok_vi()`, `_count_unigram_hits()`, `_count_phrase_hits()`
4. **Conversion** - `convert_bm25_syntax()`, `_fts5_escape_term()`

### Integration Tests (by importance)
1. **CLIP search** - Requires mock features + model
2. **BM25 search** - Requires test database
3. **Query enhancement** - Requires Gemini API or mock
4. **Video loading** - Requires test keyframes

### End-to-End Tests
1. Complete search flows (all 5 modes)
2. Result display with players
3. Video download workflow

---

## Function Statistics

- **Total functions:** 81
- **Total LOC:** ~4,459
- **Average LOC/function:** 55
- **Largest function:** `_create_base_player_html()` (361 LOC)
- **Smallest function:** `dataset_for_video()` (1 LOC)

### By Category
- Search: 18 functions (28%)
- Video: 12 functions (15%)
- Text: 10 functions (12%)
- Utils: 11 functions (14%)
- Query: 7 functions (9%)
- Player: 3 functions (4%)
- UI: 3 functions (4%)
- Server/Download: 5 functions (6%)
- Session: 4 functions (5%)
- Database: 3 functions (4%)

---

## Refactoring Impact Analysis

### High Impact Extractions (most dependencies)
1. `utils/math.py` - Used by 8+ functions
2. `utils/paths.py` - Used by 6+ functions
3. `services/search/clip_search.py` - Core logic
4. `services/database/bm25.py` - Core logic

### Low Impact Extractions (few dependencies)
1. `components/player.py` - Display only
2. `components/css.py` - Styling only
3. `services/video/downloader.py` - Isolated feature

### High Refactor Risk (complex changes)
1. Session state management (scattered across app)
2. CLIP model global state (singleton pattern needed)
3. BM25 database connections (resource management)

