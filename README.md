# Video Retrieval - CLIP + BM25 Search & Video Annotation (Streamlit App)

A comprehensive video retrieval and annotation application for the AIC25 video dataset.  
Combines **multi-aspect CLIP window search** with **BM25 transcript/OCR**, **Gemini query enhancement**, and **interactive video annotation system**.

---

## ✨ Application Features

### 🔍 **Video Search**
- **Multi-aspect CLIP Search**: Encodes multiple aspect prompts and finds the best **time windows** for each video using sliding max-pool by aspect, then aggregates with weights.
- **Transcript Reranking ("Ultra-Window")**: For top candidates, opens tight window `[t0-Δ, t1+Δ]` and counts **FTS5 hits** for **Vietnamese unigrams/phrases** derived from query + seeds, then combines `clip_n` and `text_score`.
- **BM25 Keyword Search (Independent)**: Direct transcript/OCR query using FTS5 syntax, with phrase highlighting and keyframe preview.
- **Gemini Query Enhancement (Optional)**: Converts natural language prompts into **aspects (EN prompts + weights)** and **Vietnamese seeds** for reranking step. Falls back to offline translation if no API key.
- **Video ID Search**: Direct search for specific videos by ID.

### 🎥 **Video Player**
- **YouTube Player**: Jump to matching position and display **time → frame** mapping directly using FPS from metadata CSV.
- **Kaggle Offline Player**: Download videos from Kaggle to local machine, then stream via Flask server with precise frame navigation (no size limit).
- **Dual-thumb Range Slider**: Select frame ranges with visual feedback.
- **Go to Frame**: Jump to specific frame numbers.
- **Real-time Frame Display**: Show current time and frame.

### 📝 **Video Annotation System**
- **KIS (Keyframe Identification System)**: Collect single frame annotations
- **QA (Question Answering)**: Collect frame annotations with answer text
- **TRAKE (Tracking)**: Collect multi-scene frame combinations with dynamic scene management
- **Batch Annotation**: Collect multiple frames at once with range selection
- **Data Export**: Export annotations as CSV with standard format

---

## 📦 Directory Structure & Data Contract

Set the `DATA_ROOT` environment variable to point to the directory containing two batches `b1/` and `b2/` with the following structure (minimum required fields shown):

```
PROJECT_ROOT/
├─ app.py                        # Main application file
├─ requirements.txt              # Python dependencies
├─ README_vi.md                  # Vietnamese documentation
├─ README.md                     # English documentation
├─ .env                          # Environment configuration
├─ .kaggle/                      # Kaggle API configuration
│   └─ kaggle.json              # Kaggle API token
└─ videos/                       # Downloaded videos storage
   ├─ K01_V001.mp4
   ├─ L21_V001.mp4
   └─ ...

DATA_ROOT/                       # Dataset directory (external, set via environment variable)
├─ b1/
│  ├─ clip-features-32/         # npy per video: (T, D) float32, CLIP features
│  │   ├─ L21_V001.npy
│  │   └─ ...
│  ├─ keyframes/                 # extracted keyframes (images)
│  │   └─ L21/V001/0001.jpg ...  # frame ids zero-padded (3–4 digits for B1)
│  ├─ map-keyframes/             # time mapping per video (CSV)
│  │   └─ L21_V001.csv           # required columns: pts_time (float), fps (float), n (int optional)
│  ├─ clip_keep_idxs/            # optional: indices of canonical frames
│  │   └─ L21_V001_keep_idxs.npy
│  ├─ keyframes_keepers.parquet  # optional: canonical frames table for B1
│  ├─ media-info/                # optional: JSON per video with watch_url
│  │   └─ L21_V001.json          # { "watch_url": "https://youtu.be/..." }
│  └─ text_bm25.db               # SQLite FTS5 database (see schema below)
└─ b2/
   ├─ clip-features-32/
   ├─ keyframes/                 # note: B2 typically uses 4-digit frame ids
   ├─ map-keyframes/
   ├─ clip_keep_idxs/
   ├─ manifest_dedup.parquet     # optional: includes is_canonical flag
   ├─ media-info/
   └─ text_bm25.db
```

### SQLite FTS5 (BM25) Schema

The code expects a **FTS5 virtual table** named `docs` inside `text_bm25.db` with at least these columns:

- `video_id` (TEXT) — e.g., `L21_V001`
- `text` (TEXT) — transcript/OCR text
- `t` (REAL) — timestamp (seconds)
- other columns are allowed

> The application uses:
>
> - `WHERE text MATCH ?` for searching  
> - `snippet(docs, 2, '<b>', '</b>', '...', 50)` for previews (column index `2` → `text` column)  
> - `bm25(docs)` for ranking  
> - `SELECT ... WHERE video_id=? AND t BETWEEN ? AND ?` for ultra-window counting

Ensure your SQLite build supports **FTS5**.

### `map-keyframes/*.csv`

Expected useful columns:
- `pts_time` (float) — seconds (monotonic)
- `fps` (float) — frame rate (first row is used)
- `n` (int, optional) — number of keyframes

If `n` is missing, the app estimates frames from time and `fps` plus heuristic time intervals.

---

## 🧰 Minimum Requirements

- **Python**: 3.10+ (recommended 3.11)
- **Operating System**: Windows 10/11, macOS, or Linux
- **GPU**: Not required (app only encodes CLIP text; CPU is sufficient)
- **SQLite FTS5**: Required for BM25/FTS5; Python 3.10+ usually includes FTS5. If you get FTS5 errors, use a newer Python version.
- **Data**: `DATA_ROOT` environment variable should point to directory containing `b1/` and `b2/` as described below (minimum: `clip-features-32/*.npy`, `map-keyframes/*.csv`, `text_bm25.db`). Dataset can be placed anywhere, as long as properly declared in environment variable.

## 🔧 Installation

> Recommended Python 3.10+. GPU is optional (only CLIP text encoding).

### Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Verify activation (you should see (.venv) at the beginning of the line)
```

### Install from requirements.txt
```bash
pip install -r requirements.txt
```

### Manual Installation
```bash
# Core dependencies
pip install streamlit>=1.28.0 pandas>=1.3.0 numpy>=1.21.0 pillow>=8.0.0 deep-translator>=1.9.0

# CLIP text encoder
pip install open-clip-torch>=2.20.0

# Google Gemini API (new SDK)
pip install google-genai>=0.8.0

# Additional utilities
pip install requests>=2.25.0 tqdm>=4.60.0 stopwordsiso>=0.1.0

# Kaggle Hub (for offline video player)
pip install kagglehub

# Optional: Parquet support for faster metadata loading
pip install pyarrow fastparquet
```

> **Note**: The application uses the **new** `google-genai` SDK (`google.genai.Client`) instead of `google-generativeai`.

---

## ⚙️ Configuration

You can use a `.env` file at the project root (the app reads it **without external dotenv dependencies**):

```env
DATA_ROOT=/mnt/shared/HCMC-AI/
GOOGLE_API_KEY=your_gemini_key_here
```

All env vars can also be exported normally.

### 🔑 **Kaggle API Setup**

To use the "Kaggle Offline" video player feature:

1. **Get Kaggle API Token**:
   - Login to [Kaggle](https://www.kaggle.com/)
   - Go to Account → API → Create New API Token
   - Download `kaggle.json` file to your machine

2. **Configure Kaggle API**:
   ```bash
   # Create .kaggle directory in the same code directory
   mkdir .kaggle
   
   # Copy kaggle.json file to .kaggle directory
   copy kaggle.json .kaggle\        # Windows
   cp kaggle.json .kaggle/         # Linux/macOS
   
   # Set appropriate permissions (Linux/macOS)
   chmod 600 .kaggle/kaggle.json
   ```

3. **Install Kaggle Hub**:
   ```bash
   pip install kagglehub
   ```

4. **Test connection**:
   ```python
   import kagglehub
   # Test download a small dataset to verify API
   ```

### 📁 **Video Storage Management**

**Video storage location**:
- Videos downloaded to: `./videos/` (in project directory)
- File naming format: `{video_id}.mp4` (e.g., `K01_V001.mp4`)

**Automatic cleanup**:
```bash
# Delete all downloaded videos
rm -rf ./videos/*

# Delete videos older than 7 days (Linux/macOS)
find ./videos -name "*.mp4" -mtime +7 -delete

# Delete videos older than 7 days (Windows PowerShell)
Get-ChildItem ./videos -Filter "*.mp4" | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)} | Remove-Item
```

**Storage configuration**:
- Videos are downloaded from Kaggle to local machine first, then streamed via Flask server
- `videos/` directory can grow large over time
- Recommendation: Regular cleanup or increase disk space

---

## ▶️ Running

### Activate Virtual Environment (if using)
```bash
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### Run the application
```bash
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
```

Open the local URL printed in your browser.

---

## 🖥️ Interface Overview

### 🔍 **Search Interface**

- **Search Filters**  
  - Dataset: `Both Batches / Batch 1 / Batch 2`  
  - Exclude Series: `L01,L02,K01,...`  
  - Exclude Videos: `L01_V001,...`

- **Gemini API Configuration** (optional)  
  - Enter API key, select model (`gemini-2.5-flash` or `gemini-2.5-pro`)  
  - Temperature / Top‑P sliders  
  - **Test Gemini API** (quick test)

- **Settings**  
  - `Window Second (mm:ss)` → CLIP sliding window length (default `00:40`)  
  - `Skip Intro (mm:ss)` → start scoring after this time  
  - `Rerank Window (mm:ss)` → ±Δ seconds around CLIP window for transcript reranking (default `00:20`)  
  - `Top K` → number of videos to display

- **Search Modes**
  - `CLIP Multi‑aspects Search` (no reranking)
  - `CLIP Multi‑aspects Search + Rerank` (requires Gemini to build Vietnamese seeds)
  - `Transcript Seed Search (BM25)` (search with Gemini seeds)
  - `BM25 Keyword Search` (independent text search)
  - `Search by Video ID` (search specific video)

- **Query Box**
  - Natural language (VI/EN) for CLIP modes
  - BM25 mode accepts FTS5 queries (see syntax popover in app)

### 🎯 **Video Annotation Interface**

- **Task Configuration**
  - Question Count: Number of questions to annotate
  - Task Type: Choose KIS, QA, or TRAKE
  - Answer Input: For QA tasks (when applicable)

- **Video Player Controls**
  - Frame Slider: Navigate through video frames
  - Range Slider: Select frame ranges with dual thumbs
  - Go to Frame: Jump to specific frame numbers
  - Current Button: Set range to current frame
  - Step Controls: Increase/decrease values

- **Data Collection**
  - Add Button: Collect single frame
  - Add Batch Button: Collect frame ranges
  - Scene Management: For TRAKE tasks (dynamic scene input)
  - Auto-advance: Automatically move to next scene

- **Data Management**
  - Preview Result: Live preview of collected data
  - Delete Range: Remove entries by row numbers
  - Save CSV: Export annotations with correct format
  - Notifications: Success/error feedback system

---

## 🔤 BM25 Query Syntax (FTS5)

Examples you can paste into **BM25 Keyword Search**:

```sql
cát AND điêu AND khắc
cát OR "điêu khắc"
cát NOT biển
"cột đá" AND (10 OR 12)
"công an"
```

Shortcuts recognized by the app:
- `&` → `AND`
- `|` → `OR`
- `-` → `NOT`

The app highlights matching words/phrases in full text preview.

---

## 🧠 How Scoring Works

### 1) Multi-aspect CLIP Windows
- Build multiple **aspect prompts** → encode via CLIP text encoder → normalize `L2`.
- For each video, load precomputed **frame features** `X` with shape `(T, D)` and `times` from `map-keyframes` (aligned via `keep_idxs` if available).
- For sliding window `[i..e)`, compute **max similarity** per aspect in window and **sum with weights**.
- Keep best window `win = (t0, t1)` and **peak frames per aspect** for preview.

### 2) Transcript Ultra-window Reranking
- From **Vietnamese query** + **seeds**, build:
  - `unigrams` (≤28) and `phrases` (≤18)
  - `q_any = uni ∪ phrases` and `q_phr = phrases only`
- For each candidate video, count FTS5 hits in `[t0-Δ, t1+Δ]`:
  - `hits_any = COUNT(q_any)`
  - `hits_phr = COUNT(q_phr)`
- Normalize by min-max → `clip_n`, `any_n`, `phr_n`
- `text_score = 0.3*any_n + 0.7*phr_n` (adjustable)  
- **Combined Score**: `fused = 0.4*clip_n + 0.6*text_score` (adjustable)

> Adjustable knobs exposed in code: `w_clip=0.4`, `w_text=0.6`, `text_weights = {"any": 0.3, "phr": 0.7}`, `Δ=DELTA_SEC`.

---

## 🤖 Gemini Enhancement (Aspects + Seeds)

- System prompt requests JSON:  
  - `aspects`: name (VI), `prompts` (2–4 EN), `weight` ∈ **[0.8, 1.3]**  
  - `seeds`: Vietnamese `unigrams` / `phrases`
- If API fails or no key, **fallback** translator builds minimal aspects from line-by-line translation and empty seeds.

> In **basic CLIP mode**, you can still **manually adjust aspect weights** via sliders (0.5 → 1.6) to rerank without re-encoding.

---

## 🎬 YouTube Player Helper

If JSON exists at `media-info/{video_id}.json` with `{ "watch_url": "..." }`, the app renders player:
- Seek to matching seconds.
- Display **current time** and **frame** calculated via `FPS` read from `map-keyframes/{video_id}.csv`.
- Allow seeking by **frame number**.

## 🎬 Kaggle Offline Player with Flask Streaming

The application uses Flask server to stream videos from local storage:
- **Download Process**: Videos are downloaded from Kaggle to `./videos/` first
- **Flask Streaming**: Flask server runs on port 8801 to stream videos
- **No Size Limit**: No 200MB limit from Kaggle API
- **Frame Navigation**: Precise frame navigation with FPS from metadata
- **Auto-start**: Video starts at specified timestamp

---

## 🎯 Video Annotation Features

The application includes comprehensive video annotation capabilities for data collection and labeling tasks:

### 📝 **Task Types**

- **KIS (Keyframe Identification System)**: Collect single frame annotations
- **QA (Question Answering)**: Collect frame annotations with answer text
- **TRAKE (Tracking)**: Collect multi-scene frame combinations with dynamic scene management

### 🎮 **Interactive Video Player**

- **Dual-thumb Range Slider**: Select frame ranges with visual feedback
- **Frame Slider**: Navigate through video frames
- **Go to Frame**: Jump to specific frame numbers
- **Real-time Display**: Show current time in `mm:ss:ff` format
- **Auto-start**: Video starts at specified timestamp

### 📊 **Data Collection Interface**

- **Question Count Input**: Specify number of questions to annotate
- **Task Selection**: Choose between KIS, QA, or TRAKE tasks
- **Answer Input**: For QA tasks, enter answer text
- **Scene Management**: For TRAKE tasks, manage multiple scenes with dynamic input

### 🎨 **Advanced UI Features**

- **Visual Scene Selection**: Scene buttons with color-coded status (red=empty, green=filled)
- **Auto-advance**: Automatically move to next scene after adding data
- **Current Button**: Set range boundaries to current frame
- **Step Controls**: Increase/decrease range values
- **Real-time Validation**: Input validation with error notifications

### 💾 **Data Management**

- **Live Preview**: Real-time preview of collected data (limit: 100 entries)
- **Range Deletion**: Delete entries by specifying row ranges
- **CSV Export**: Save annotations with `query-<count>-<task>.csv` format
- **Error Handling**: Comprehensive validation and error notifications
- **Success Feedback**: Confirmation notifications for all operations

### 📋 **Usage Workflow**

1. **Setup**: Enter question count and select task type
2. **Navigate**: Use sliders and controls to find desired frames
3. **Collect**: Add single frames or ranges using Add/Add Batch buttons
4. **Manage**: Review, edit, or delete collected data in Preview Result
5. **Export**: Save final annotations as CSV file

---

## 🛠️ Troubleshooting

### **Data and Configuration**
- **No results / missing data**  
  - Check `DATA_ROOT` and ensure all required directories exist.  
  - Verify `clip-features-32/{video_id}.npy` and `map-keyframes/{video_id}.csv` are present.
  - Check file permissions and paths containing special characters.
- **Virtual Environment Issues**
  - Ensure virtual environment is activated: `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
  - Check Python path: `which python` (Linux/macOS) or `where python` (Windows)
  - If import errors occur, try reinstalling: `pip install -r requirements.txt`

### **BM25 Search**
- **BM25 errors**  
  - Confirm `text_bm25.db` exists for selected batch and `docs` is **FTS5**.  
  - Ensure `text` column is at index `2` for `snippet(docs, 2, ...)` or adjust column index.
  - Check SQLite version supports FTS5: `sqlite3 --version`
  - Test simple query: `"công an"` or `cát AND điêu`

### **Gemini API**
- **Gemini errors**  
  - Provide valid `GOOGLE_API_KEY` from https://aistudio.google.com
  - Use new SDK: `google-genai>=0.8.0` (not `google-generativeai`)
  - Test API connection using "Test Gemini API" button in app
  - Check API quota and rate limits

### **Video and Images**
- **Images not found**  
  - App tries fallbacks (±N frames, canonical list, or directory scan)
  - Check zero-padding width per batch (B1 uses 3–4, B2 typically 4)
  - Verify `keyframes/{video_id}/` contains image files
- **Video won't play**
  - YouTube: Check valid URL and video not restricted
  - Kaggle Offline: Ensure Kaggle dataset access
  - Local server: Check port 8801 not occupied
  - Flask streaming: Check Flask server running on port 8801
- **Kaggle API Issues**
  - Check `kaggle.json` file in `./.kaggle/` (in project directory)
  - Test API with Python: `import kagglehub`
  - Ensure dataset terms accepted on Kaggle website
  - Check Kaggle API quota and rate limits
  - Check internet connection and firewall settings
- **Video Storage Issues**
  - Check available disk space
  - Delete old videos: `rm -rf ./videos/*` (Linux/macOS) or `Remove-Item ./videos/*` (Windows)
  - Check write permissions to `./videos/` directory

### **Video Annotation**
- **Annotation errors**
  - Ensure valid question count (> 0)
  - For QA tasks: enter answer before adding frames
  - For TRAKE tasks: enter scene count and frames per scene
  - Check 100 entry limit in preview

### **Performance**
- **App runs slowly**
  - Reduce `Top K` to display fewer videos
  - Increase `Skip Intro` to skip video beginning
  - Use GPU if available for CLIP encoding
  - Check RAM and CPU usage

---

## 🧪 Minimal Smoke Testing

### **Basic Tests**
1) **BM25 Keyword Search**: Query `"công an"` to validate FTS5
2) **CLIP Multi-aspects Search**: Enter short VI description (no Gemini needed)
3) **Video ID Search**: Find `K01_V001` or `L21_V001`

### **Advanced Tests**
4) **Gemini Integration**: Enter API key and test "Enhance Query"
5) **Rerank**: Run "CLIP Multi-aspects Search + Rerank"
6) **Video Annotation**: Test annotation with KIS/QA/TRAKE tasks

### **Video Player Tests**
7) **YouTube Player**: Click "Watch Video" → "Youtube Online"
8) **Kaggle Offline**: Click "Watch Video" → "Kaggle Offline"

---

## 📚 Usage Examples

### **Basic Search**
```bash
# BM25 queries
"công an"                    # Exact phrase
cát AND điêu AND khắc        # All words required
cát OR "điêu khắc"           # Either word or phrase
cát NOT biển                 # "cát" without "biển"

# CLIP queries (Vietnamese/English)
"video về điêu khắc cát"
"video showing sand sculpture"
"bài học về nghệ thuật"
```

### **Advanced Configuration**
```bash
# Environment variables
DATA_ROOT=/path/to/HCMC-AI/
GOOGLE_API_KEY=your_gemini_key_here

# Streamlit options
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
```

### **Annotation Workflow**
1. **Setup**: Choose task (KIS/QA/TRAKE), enter question count
2. **Navigate**: Use frame slider to find desired position
3. **Collect**: Add single frame or Add Batch range
4. **Review**: Check Preview Result, delete if needed
5. **Export**: Save CSV with standard format

---

## 📜 License

This README documents how to run the provided application. Use at your own risk. Dataset license and video links must follow their original terms.
