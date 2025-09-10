# HCMC AI Competition – Visual + Text Search App

A Streamlit app for multi-modal video search using:
- CLIP Multi-aspects Visual Search
- Optional CLIP + Transcript Rerank (Gemini-assisted aspects/seeds)
- BM25 Keyword Search over transcripts

This README covers setup, running, features, and troubleshooting.

## 1) Prerequisites
- Python 3.10+ (tested with 3.12)
- Pip, virtual environment recommended
- Optional: NVIDIA GPU + CUDA for faster CLIP inference

## 2) Project Setup

### 2.1 Create .env
Create a `.env` in project root (or copy from `env_config.txt`):

```env
# Root path to the dataset (use forward slashes)
DATA_ROOT=E:/HCMC-AI/

# Optional: Google Gemini API key (for Enhance Query + Rerank)
GOOGLE_API_KEY=your_api_key_here
```

Notes:
- Use forward slashes `/` on Windows
- `DATA_ROOT` must end with `/`

### 2.2 Install dependencies
```bash
pip install -r requirements.txt
```

If needed manually:
```bash
pip install streamlit numpy pandas Pillow deep-translator torch torchvision
pip install open-clip-torch clip-by-openai google-genai
```

## 3) Data directory structure
Ensure the dataset exists under `DATA_ROOT` with the following structure. Items marked (required) are needed for core functionality; items marked (optional) enable quality improvements.

```
E:/HCMC-AI/
├── b1/
│   ├── clip-features-32/                 # (required) *.npy CLIP feature files: <VIDEO_ID>.npy
│   ├── map-keyframes/                    # (required) *.csv mapping pts_time → keyframe index per video
│   │   └── <VIDEO_ID>.csv
│   ├── keyframes/                        # (required) extracted keyframe images
│   │   └── Lxx/ Vxxx/ 0001.jpg ...       # zero-padded filenames
│   ├── video/                            # (optional) original videos if needed
│   ├── clip_keep_idxs/                   # (optional) *_keep_idxs.npy for subsampling alignment
│   │   └── <VIDEO_ID>_keep_idxs.npy
│   ├── text_bm25.db                      # (required for BM25) SQLite FTS5 database
│   ├── media-info/                       # (optional) <VIDEO_ID>.json with { "watch_url": ... }
│   │   └── <VIDEO_ID>.json
│   ├── keyframes_keepers.parquet         # (optional) canonical keyframes metadata (duplicate filtering)
│   ├── duplicates.parquet                # (optional) duplicate info (if available)
│   └── phash.parquet                     # (optional) perceptual hash info (if available)
└── b2/
    ├── clip-features-32/                 # (required) *.npy CLIP feature files
    ├── map-keyframes/                    # (required) *.csv mapping pts_time → keyframe index per video
    ├── keyframes/                        # (required) extracted keyframe images
    ├── video/                            # (optional)
    ├── clip_keep_idxs/                   # (optional)
    ├── text_bm25.db                      # (required for BM25)
    ├── media-info/                       # (optional)
    └── manifest_dedup.parquet            # (optional) canonical keyframes for B2 (deduped)
```

### 3.1 File details and usage
- `clip-features-32/<VIDEO_ID>.npy` (required): CLIP features matrix [N×D] used for windowed similarity.
- `map-keyframes/<VIDEO_ID>.csv` (required): includes at least `pts_time` (and optionally `fps`, `n`); used to map seconds → keyframe index and compute FPS.
- `keyframes/` (required): keyframe images referenced by selected frames (zero-padded names, e.g., 0001.jpg). For B2, canonical filtering may apply.
- `clip_keep_idxs/<VIDEO_ID>_keep_idxs.npy` (optional): index mapping aligning features to CSV times.
- `text_bm25.db` (required for BM25): SQLite FTS5 database with `docs(video_id, text, t, ...)` for transcript search.
- `media-info/<VIDEO_ID>.json` (optional): `{"watch_url": "https://www.youtube.com/watch?v=..."}` to embed a player.
- `keyframes_keepers.parquet` (B1, optional): canonical frames metadata used to skip duplicates.
- `manifest_dedup.parquet` (B2, optional): contains `is_canonical` to filter duplicates.
- `duplicates.parquet`, `phash.parquet` (optional): additional duplicate/phash metadata (used if present).

## 4) Run the app

### Option A: Windows batch
```bash
run_streamlit.bat
```

### Option B: Direct command
```bash
cd E:/HCMC-AI-Competition
streamlit run streamlit_app.py --server.port 8501
```
If port 8501 is busy:
```bash
streamlit run streamlit_app.py --server.port 8502
```

## 5) Features and Modes

### 5.1 Search Filters
- Dataset Selection: Both / Batch 1 / Batch 2
- Exclude Series: e.g. `L01,L02,K01`
- Exclude Videos: e.g. `L01V001,L01V002`

Filters apply to both CLIP and BM25.

### 5.2 Modes
- CLIP Multi-aspects Search
- CLIP Multi-aspects Search + Rerank (requires Gemini)
- BM25 Keyword Search

### 5.3 Turn off Gemini (checkbox)
- Location: below “Max aspects” and above the query box
- When ON:
  - “Enhance Query” is hidden; only “Run Query” is shown
  - Rerank mode is blocked. If selected, app shows a warning and runs basic CLIP search
  - App uses fallback translation for aspects/seeds (no Gemini calls)
  - Logs will explicitly show: “Using fallback translation method (Gemini off or not configured)”

### 5.4 Enhance Query and Run workflow (CLIP + Gemini)
- First time click “Enhance Query”: app calls Gemini to build aspects + seeds
- After a successful enhance:
  - App auto-runs once to show results
  - UI shows both buttons: “Run Query” and “Reset Query”
  - You can reuse the enhanced aspects/seeds (cached) to run again or to switch to rerank mode
- When enhanced data is cached, the app will NOT call fallback translation
 - If Gemini fails, logs show: “Query enhance failed — using fallback translation | error: <message>”

## 6) BM25 Keyword Search
- Runs directly on transcripts using SQLite FTS5
- Operators: AND, OR, NOT, and phrase search with quotes

Examples:
```sql
cát AND điêu AND khắc
cát OR "điêu khắc"
cát NOT biển
"điêu khắc cát"
```

Behavior:
- Timestamps are mapped to keyframes using `map-keyframes/{video_id}.csv`
- Keyframes are shown with duplicate filtering (B1 `keyframes_keepers.parquet`; B2 `manifest_dedup.parquet`)
- Full text is displayed with highlighted keywords

## 7) CLIP Visual Search
- Multi-aspect scoring over windows
- Window score = sum over aspects of weight × max(similarity) in the window
- Displays best window and per-aspect keyframes

## 8) Rerank with Transcript (requires Gemini)
- After CLIP ranking, top candidates are reranked by transcript hits
- Requires Vietnamese seeds from Gemini
- If Gemini is turned off or not configured, rerank is unavailable

## 9) Troubleshooting
- Port already in use:
  - Use another port: `--server.port 8502`
- No results:
  - Check `DATA_ROOT` and directory structure
  - Ensure `.npy` features and `text_bm25.db` exist
- Gemini not working:
  - Verify `GOOGLE_API_KEY` in `.env`
  - Use the “🧪 Test Gemini API” button in the app
- Performance:
  - Enable GPU (install CUDA-capable torch build)

## 10) Notes
- Paths use forward slashes for Windows compatibility
- `.env` should be UTF-8 (no BOM)
- `DATA_ROOT` must end with `/`

---
This app is tailored for HCMC AI Competition datasets and workflows.
