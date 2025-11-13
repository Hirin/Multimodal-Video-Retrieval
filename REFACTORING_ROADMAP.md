# Refactoring Roadmap: Modular Structure

## Target Architecture

```
multimodal-video-retrieval/
├── app.py (streamlit main entry point - 200 LOC)
├── config/
│   ├── __init__.py
│   ├── default.py (default parameters)
│   ├── db.py (database paths)
│   ├── paths.py (data root, batch paths)
│   ├── gemini.py (LLM configuration)
│   └── video.py (video server configuration)
│
├── models/
│   ├── __init__.py
│   ├── video.py (Video, Batch, VideoMetadata)
│   ├── search.py (SearchQuery, SearchResult, Aspects, Seeds)
│   ├── clip.py (CLIPFeatures, FrameData)
│   └── bm25.py (BM25Result, HighlightedText)
│
├── services/
│   ├── __init__.py
│   ├── search/
│   │   ├── __init__.py
│   │   ├── clip_search.py (CLIP ranking)
│   │   ├── bm25_search.py (BM25 keyword search)
│   │   ├── seed_search.py (Transcript seed search)
│   │   └── rerank.py (Reranking with text)
│   │
│   ├── query/
│   │   ├── __init__.py
│   │   ├── enhancer.py (Gemini enhancement)
│   │   ├── seed_extractor.py (Seed extraction)
│   │   ├── fallback.py (Translation fallback)
│   │   └── clip_encoder.py (CLIP encoding)
│   │
│   ├── video/
│   │   ├── __init__.py
│   │   ├── loader.py (Keyframes, features)
│   │   ├── metadata.py (Video info, FPS)
│   │   ├── downloader.py (Kaggle downloads)
│   │   └── server.py (Flask video server)
│   │
│   └── database/
│       ├── __init__.py
│       ├── bm25.py (FTS5 queries)
│       ├── keyframe.py (Keyframe metadata)
│       └── connection.py (DB connections)
│
├── components/
│   ├── __init__.py
│   ├── sidebar.py (Configuration UI)
│   ├── search_input.py (Query inputs)
│   ├── results.py (Result display)
│   ├── player.py (Video player HTML)
│   ├── download.py (Download dialog)
│   ├── css.py (Styling)
│   └── animations.py (Loading/UI animations)
│
├── utils/
│   ├── __init__.py
│   ├── math.py (Normalization, conversion)
│   ├── paths.py (Path resolution)
│   ├── text.py (Vietnamese processing)
│   ├── types.py (Type conversions)
│   ├── logging.py (Function logging)
│   └── time.py (Time utilities)
│
├── state/
│   ├── __init__.py
│   └── session.py (Session state management)
│
├── exceptions/
│   ├── __init__.py
│   ├── search.py (Search-related)
│   ├── video.py (Video-related)
│   ├── database.py (Database-related)
│   └── api.py (API-related)
│
└── tests/
    ├── __init__.py
    ├── test_search/
    ├── test_utils/
    ├── test_video/
    └── test_database/
```

---

## Phase 1: Extract Core Services (Week 1-2)

### Goal: Separate business logic from UI

#### 1.1 Create `services/search/clip_search.py`
**From app.py lines:** 1138-1245

**Functions to move:**
- `build_query_vecs_for_aspects()` (1138)
- `best_window_multi_aspect()` (1156)
- `score_video_clip_window()` (1201)
- `rank_videos_clip_window()` (1215)

**Dependencies:**
- `l2_normalize_rows()` → utils/math.py
- `encode_text()` → services/query/clip_encoder.py
- `npy_path_for_video()` → utils/paths.py
- `load_times_for_video()` → services/video/loader.py
- `list_videos_from_dirs()` → services/video/loader.py

**Tests:**
- Test vector encoding
- Test window scoring
- Test ranking with mock features

---

#### 1.2 Create `services/search/bm25_search.py`
**From app.py lines:** 1547-1650

**Functions to move:**
- `bm25_search_videos()` (1547)
- `convert_bm25_syntax()` (1381)
- `build_keyword_sets()` (1251)
- `highlight_keywords_in_text()` (1297)
- `apply_search_filters()` (1471)
- `deduplicate_bm25_results()` (1499)

**Dependencies:**
- `_tok_vi()` → utils/text.py
- `_connect_ro()` → services/database/connection.py
- `highlight_keywords_in_text()` → utils/text.py
- `_count_unigram_hits()` → utils/text.py
- `_count_phrase_hits()` → utils/text.py

**Tests:**
- Test query conversion
- Test keyword highlighting
- Test deduplication logic
- Mock database queries

---

#### 1.3 Create `services/query/clip_encoder.py`
**From app.py lines:** 979-1152

**Functions to move:**
- `ensure_clip_text_encoder()` (979)
- `encode_text()` (1001)

**Considerations:**
- Global CLIP model state management
- Singleton pattern for model loading
- Error handling for missing models

**Tests:**
- Test model loading
- Test encoding consistency
- Test vector normalization

---

#### 1.4 Create `services/query/enhancer.py`
**From app.py lines:** 1043-1135

**Functions to move:**
- `extract_aspects_and_seeds_with_gemini()` (1043)

**Dependencies:**
- Google Gemini API
- Fallback translator

**Error Handling:**
- API failures → fallback
- Malformed responses → validation
- Rate limiting → retry logic

---

#### 1.5 Create `utils/math.py`
**From app.py lines:** 465-495

**Functions to move:**
- `l2_normalize_rows()` (465)
- `sec_to_mmss()` (470)
- `parse_mmss()` (479)
- `_as_int()` (947)
- `_as_float()` (963)

**Tests:**
- Test normalization
- Test time conversions
- Test type safety

---

#### 1.6 Create `utils/paths.py`
**From app.py lines:** 496-535

**Functions to move:**
- `dataset_for_video()` (496)
- `keyframe_path()` (499)
- `npy_path_for_video()` (514)
- `map_csv_path_for_video()` (518)
- `keep_idxs_path_for_video()` (522)
- `safe_imread()` (526)

**Tests:**
- Test path generation for both batches
- Test file existence checks
- Test image loading fallbacks

---

### Checklist: Phase 1

- [ ] Create `services/search/clip_search.py`
- [ ] Create `services/search/bm25_search.py`
- [ ] Create `services/query/clip_encoder.py`
- [ ] Create `services/query/enhancer.py`
- [ ] Create `utils/math.py`
- [ ] Create `utils/paths.py`
- [ ] Update app.py imports
- [ ] Run tests for each module
- [ ] Update session state management
- [ ] Verify search modes still work

---

## Phase 2: Extract Video & Database Services (Week 2-3)

### Goal: Separate data access layer

#### 2.1 Create `services/video/loader.py`
**From app.py lines:** 536-726

**Functions to move:**
- `timestamp_to_keyframe_number()` (536)
- `find_best_keyframe()` (575)
- `list_videos_from_dirs()` (693)
- `load_times_for_video()` (720)

**Dependencies:**
- Path utilities
- Image loading
- Pandas for CSV reading

---

#### 2.2 Create `services/video/metadata.py`
**From app.py lines:** 2565-2606

**Functions to move:**
- `get_fps_from_csv()` (2565)
- `get_watch_url()` (2587)
- `get_video_info()` (2606)

---

#### 2.3 Create `services/database/bm25.py`
**From app.py lines:** 1273-1791

**Functions to move:**
- `_connect_ro()` (1273)
- `_count_one()` (1276)
- `_alias_windows_and_dbpaths()` (1288)
- `_run_fts5_query()` (1791)
- `_resolve_bm25_dbpaths()` (1859)
- `transcript_seed_bm25_search()` (1888)

**Class Structure:**
```python
class BM25Database:
    def __init__(self, db_path):
        self.db_path = db_path

    def search(self, query, limit):
        pass

    def count_matches(self, video_id, t0, t1, query):
        pass
```

---

#### 2.4 Create `services/database/keyframe.py`
**From app.py lines:** 1401-1499

**Functions to move:**
- `load_keyframes_data()` (1401)
- `get_keyframes_data()` (1464)

**Caching:**
- Implement `_KEYFRAMES_CACHE` as class variable
- Add cache invalidation methods

---

#### 2.5 Create `services/video/downloader.py`
**From app.py lines:** 147-358

**Functions to move:**
- `download_video_dialog()` (147)
- `download_video_from_kaggle()` (326)

**Dependencies:**
- Kaggle API
- Dialog state management

---

#### 2.6 Create `services/video/server.py`
**From app.py lines:** 358-420

**Functions to move:**
- `make_video_app()` (358)
- `ensure_server_once()` (378)

**Architecture:**
```python
class VideoServer:
    _instance = None

    def __init__(self, root_dir, host, port):
        pass

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls(...)
        return cls._instance
```

---

### Checklist: Phase 2

- [ ] Create `services/video/loader.py`
- [ ] Create `services/video/metadata.py`
- [ ] Create `services/database/bm25.py`
- [ ] Create `services/database/keyframe.py`
- [ ] Create `services/video/downloader.py`
- [ ] Create `services/video/server.py`
- [ ] Create `services/database/connection.py` (shared DB utilities)
- [ ] Implement proper error handling
- [ ] Add logging
- [ ] Test all database operations

---

## Phase 3: Extract Configuration (Week 3)

### Goal: Centralize configuration management

#### 3.1 Create `config/default.py`
```python
# Search parameters
WINDOW_SEC = 40.0
SKIP_SEC = 0.0
DELTA_SEC = 20.0
TOP_VIDEOS = 20
DEFAULT_TOP_K = 20
LIMIT_VIDEOS = None
```

#### 3.2 Create `config/paths.py`
```python
DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/shared/HCMC-AI/")
LOCAL_VIDEO_ROOT = Path("./videos")
LOCAL_VIDEO_URL_BASE = "http://127.0.0.1:8801/video"

# Batch definitions
BATCH_CONFIGS = {
    "b1": {...},
    "b2": {...}
}
```

#### 3.3 Create `config/gemini.py`
```python
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MAX_TOKENS = 10000
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
```

#### 3.4 Create `config/db.py`
```python
def find_db_path(batch_name):
    pass

B1_DB_PATH = find_db_path("b1")
B2_DB_PATH = find_db_path("b2")
```

---

## Phase 4: Extract UI Components (Week 4)

### Goal: Separate UI logic from business logic

#### 4.1 Create `components/sidebar.py`
- Search mode selector
- Dataset filters
- Gemini configuration
- Search parameters (Window, Skip, Delta, Top-K)
- Debug toggle

**Function:** `render_sidebar()` → Returns session state updates

#### 4.2 Create `components/search_input.py`
- Query text area
- Video ID input
- Max aspects input
- BM25 syntax help

#### 4.3 Create `components/results.py`
- BM25 results display
- CLIP results display
- Rerank results display
- Result pagination

#### 4.4 Create `components/player.py`
- `_create_base_player_html()`
- `create_youtube_player_html()`
- `create_local_player_html()`

#### 4.5 Create `components/download.py`
- `render_download_dialog()`

#### 4.6 Create `components/css.py`
- `create_popover_css()`
- `create_button_styles()`
- `create_result_styles()`

#### 4.7 Create `components/animations.py`
- `create_loading_overlay()`
- `create_spinner()`

---

## Phase 5: Extract Utilities (Week 4)

### Goal: Centralize utility functions

#### 5.1 Create `utils/text.py`
- `_tok_vi()` (1248)
- `_vi_strip_accents()` (1752)
- `_fts5_escape_term()` (1755)
- `_count_unigram_hits()` (1821)
- `_count_phrase_hits()` (1843)
- `_build_seed_fts5_query()` (1769)
- `highlight_keywords_in_text()` (1297)

#### 5.2 Create `utils/types.py`
- `_as_int()`
- `_as_float()`

#### 5.3 Create `utils/time.py`
- `_window_key()`

#### 5.4 Create `utils/logging.py`
- `run_with_logs()`

---

## Phase 6: Refactor Main App.py (Week 5)

### Goal: Reduce app.py to pure UI orchestration

**New app.py structure (~200 LOC):**

```python
import streamlit as st
from config import default
from components import sidebar, search_input, results
from services.search import (
    clip_search, bm25_search, seed_search, rerank
)
from services.query import enhancer, clip_encoder
from state import session

# Page config
st.set_page_config(...)

# Initialize session
session.initialize()

# Load CLIP model
clip_encoder.ensure_clip_text_encoder()

# Render sidebar
sidebar.render()

# Render main content
st.title("Video Retrieval")

# Render search inputs
search_input.render()

# Handle search execution
if st.session_state.run_flag:
    results.render()

# Debug panel
if st.session_state.get("debug_mode"):
    st.write("Debug info...")
```

---

## Phase 7: Add Tests (Week 5-6)

### Test Structure

```python
# tests/test_search/test_clip_search.py
def test_best_window_multi_aspect():
    # Mock features and query vectors
    # Test window finding algorithm
    pass

# tests/test_utils/test_math.py
def test_l2_normalize():
    # Test normalization
    pass

# tests/test_database/test_bm25.py
def test_bm25_search():
    # Use test database
    pass
```

---

## Migration Strategy

### Step 1: Create all service modules in parallel (don't break app.py yet)
- Add imports from new modules where possible
- Keep old functions as wrappers calling new modules

### Step 2: Update imports incrementally
- First, update internal function calls
- Test each service independently

### Step 3: Remove old functions from app.py
- One by one, migrate functions
- Verify tests pass after each removal

### Step 4: Refactor UI components
- Extract UI functions last
- Minimize coupling between components

### Step 5: Final cleanup
- Remove duplicate imports
- Fix circular dependencies
- Optimize module organization

---

## Metrics to Track

### Code Quality
- [ ] Reduce app.py from 4459 to <500 lines
- [ ] Create 15+ service modules
- [ ] Increase test coverage from 0% to >70%
- [ ] Reduce cyclomatic complexity per function

### Performance
- [ ] Module import time <100ms
- [ ] CLIP model loading cached
- [ ] Database queries optimized
- [ ] No memory leaks from circular references

### Maintainability
- [ ] Clear separation of concerns
- [ ] Each module has single responsibility
- [ ] Documented API for each service
- [ ] Examples for common usage patterns

---

## Dependencies Between Modules

```
app.py
├─ config/ (no dependencies)
├─ models/ (no dependencies)
├─ services/
│  ├─ search/ (depends on: services/query, services/video, services/database, utils/)
│  ├─ query/ (depends on: utils/, config/)
│  ├─ video/ (depends on: utils/, config/)
│  └─ database/ (depends on: config/, utils/)
├─ components/ (depends on: services/, models/)
├─ utils/ (no dependencies)
├─ state/ (depends on: config/)
└─ exceptions/ (no dependencies)
```

---

## Priority Functions to Extract First

**Must Extract (Used in multiple places):**
1. CLIP search functions (duplicated logic)
2. Path utilities (used everywhere)
3. Math utilities (used in multiple services)
4. Database connection functions (shared resource)

**Extract Early (Heavy lifting):**
5. Gemini integration (complex logic)
6. BM25 search (complex queries)
7. Video loading (heavy I/O)

**Extract Later (UI-heavy):**
8. Components (mostly display logic)
9. Session management (lots of state)

---

## Success Criteria

- [ ] All search modes work identically post-refactor
- [ ] No performance degradation
- [ ] Test coverage >70%
- [ ] Code duplication <5%
- [ ] Each module <500 lines
- [ ] Clear API documentation
- [ ] Streamlit UI unchanged from user perspective

