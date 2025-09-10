# HCMC AI 2025 – CLIP + BM25 Search (Streamlit App)

A fast, no‑frills retrieval UI for the AIC25 / HCMC AI 2025 video dataset.  
It fuses **CLIP multi‑aspect window search** with **BM25 transcript/OCR** and an optional **Gemini‑powered query expander**.

---

## ✨ What this app does

- **CLIP Multi‑Aspect Search**: Encodes several aspect prompts and finds the best **time window** per video using a sliding max‑pool per‑aspect, then sums with weights.
- **Transcript Rerank (“Ultra‑Window”)**: For the top candidates, opens a tight window `[t0-Δ, t1+Δ]` and counts **FTS5 hits** for Vietnamese **unigrams/phrases** derived from your query + seeds, then fuses `clip_n` and `text_score`.
- **BM25 Keyword Search (standalone)**: Query the transcript/OCR directly using FTS5 syntax, with phrase highlighting and keyframe previews.
- **Gemini Query Enhancer (optional)**: Turns a natural-language prompt into **aspects (EN prompts + weights)** and **Vietnamese seeds** for the rerank step. Falls back to offline translation if no API key.
- **YouTube Player Overlay**: Jumps to the matched spot and displays live **time → frame** mapping using FPS from your metadata CSV.

---

## 📦 Directory layout & data contracts

Set `DATA_ROOT` to a folder containing two batches `b1/` and `b2/` with the following structure (minimum viable fields shown):

```
DATA_ROOT/
├─ b1/
│  ├─ clip-features-32/         # npy per video:  (T, D) float32, CLIP features
│  │   ├─ L21_V001.npy
│  │   └─ ...
│  ├─ keyframes/                 # extracted keyframes (images)
│  │   └─ L21/V001/0001.jpg ...  # zero-padded frame ids (3–4 digits for B1)
│  ├─ map-keyframes/             # timing map per video (CSV)
│  │   └─ L21_V001.csv           # required columns: pts_time (float), fps (float), n (int optional)
│  ├─ clip_keep_idxs/            # optional: indices of canonical frames
│  │   └─ L21_V001_keep_idxs.npy
│  ├─ keyframes_keepers.parquet  # optional: canonical frames table for B1
│  ├─ media-info/                # optional: per-video JSON with watch_url
│  │   └─ L21_V001.json          # { "watch_url": "https://youtu.be/..." }
│  └─ text_bm25.db               # SQLite FTS5 database (see schema below)
└─ b2/
   ├─ clip-features-32/
   ├─ keyframes/                 # note: B2 tends to use 4‑digit frame ids
   ├─ map-keyframes/
   ├─ clip_keep_idxs/
   ├─ manifest_dedup.parquet     # optional: include is_canonical flag
   ├─ media-info/
   └─ text_bm25.db
```

### SQLite FTS5 (BM25) schema

The code expects an **FTS5 virtual table** named `docs` inside `text_bm25.db` with at least these columns:

- `video_id` (TEXT) — e.g., `L21_V001`
- `text` (TEXT) — transcript/OCR text
- `t` (REAL) — timestamp (seconds)
- other columns allowed

> The app uses:
>
> - `WHERE text MATCH ?` for searching  
> - `snippet(docs, 2, '<b>', '</b>', '...', 50)` for previews (column index `2` → the `text` column)  
> - `bm25(docs)` for ranking  
> - `SELECT ... WHERE video_id=? AND t BETWEEN ? AND ?` for ultra‑window counting

Make sure your SQLite build supports **FTS5**.

### `map-keyframes/*.csv`

Expected useful columns:
- `pts_time` (float) — seconds (monotonic)
- `fps` (float) — frame rate (first row is used)
- `n` (int, optional) — keyframe number

If `n` is missing, the app estimates frames from time and `fps` plus a heuristic interval.

---

## 🔧 Installation

> Python 3.10+ recommended. GPU is optional (CLIP text encodes only).

```bash
# Core
pip install streamlit pandas numpy pillow deep-translator

# CLIP text encoder (either open-clip or the original CLIP)
pip install open-clip-torch          # preferred
# Optional alternative:
# pip install git+https://github.com/openai/CLIP.git

# Gemini (two SDKs are referenced by the code)
pip install google-genai>=0.3.0      # used in main extractor (Client API)
pip install google-generativeai>=0.7 # used in the small "Test API" panel

# Parquet (optional but recommended for fast metadata)
pip install pyarrow fastparquet
```

> If you only want the **new** `google-genai` SDK, either install both packages as above or update the “Test Gemini API” block to also use `google.genai.Client` for consistency.

---

## ⚙️ Configuration

You can use a `.env` file at the project root (the app reads it **without external dotenv deps**):

```env
DATA_ROOT=/mnt/shared/HCMC-AI/
GOOGLE_API_KEY=your_gemini_key_here
```

All env vars can also be exported normally.

---

## ▶️ Run

Save your code as `app.py` (or any name) and run:

```bash
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
```

Open the printed local URL in your browser.

---

## 🖥️ UI overview

- **Search Filters**  
  - Dataset: `Both / Batch 1 / Batch 2`  
  - Exclude Series: `L01,L02,K01,...`  
  - Exclude Videos: `L01_V001,...`

- **Gemini API Configuration** (optional)  
  - Enter API key, pick model (`gemini-2.5-flash` or `gemini-2.5-pro`)  
  - Temperature / Top‑P sliders  
  - **Test Gemini API** (quick sanity check)

- **Setting**  
  - `Window Second (mm:ss)` → CLIP sliding window length (default `00:40`)  
  - `Skip Intro (mm:ss)` → start scoring after this time  
  - `Rerank Window (mm:ss)` → ±Δ seconds around the CLIP window for transcript rerank (default `00:20`)  
  - `Top K` → number of videos to display

- **Search Mode**
  - `CLIP Multi‑aspects Search` (no rerank)
  - `CLIP Multi‑aspects Search + Rerank` (needs Gemini to build Vietnamese seeds)
  - `BM25 Keyword Search` (standalone text search)

- **Query box**
  - Natural language (VI/EN) for CLIP modes
  - BM25 mode accepts FTS5 queries (see syntax popover in the app)

---

## 🔤 BM25 query syntax (FTS5)

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

The app highlights matched words/phrases in the full text preview.

---

## 🧠 How scoring works

### 1) CLIP multi‑aspect window
- Build multiple **aspect prompts** → encode via CLIP text encoder → `L2` normalize.
- For each video, load precomputed **frame features** `X` of shape `(T, D)` and `times` from `map-keyframes` (aligned via `keep_idxs` if present).
- For a sliding window `[i..e)`, compute the **max similarity** per aspect within the window and **sum with weights**.
- Keep the best window `win = (t0, t1)` and the **peak frames per aspect** for preview.

### 2) Transcript ultra‑window rerank
- From the **Vietnamese query** + **seeds**, build:
  - `unigrams` (≤28) and `phrases` (≤18)
  - `q_any = uni ∪ phrases` and `q_phr = phrases only`
- For each candidate video, count FTS5 hits within `[t0-Δ, t1+Δ]`:
  - `hits_any = COUNT(q_any)`
  - `hits_phr = COUNT(q_phr)`
- Normalize by min‑max → `clip_n`, `any_n`, `phr_n`
- `text_score = 0.5*any_n + 0.5*phr_n` (tunable)  
- **Fused score**: `fused = 0.6*clip_n + 0.4*text_score` (tunable)

> Knobs exposed in code: `w_clip`, `w_text`, `text_weights = {"any": a, "phr": b}`, `Δ=DELTA_SEC`.

---

## 🤖 Gemini expander (aspects + seeds)

- System instruction requests JSON:  
  - `aspects`: name (VI), `prompts` (2–4 EN), `weight` ∈ **[0.8, 1.3]**  
  - `seeds`: Vietnamese `unigrams` / `phrases`
- If the API fails or no key is set, a **fallback** translator builds minimal aspects from per‑line translations and empty seeds.

> In **basic CLIP mode**, you can still **manually adjust aspect weights** via sliders (0.5 → 1.6) to reshuffle the ranking without re‑encoding.

---

## 🎬 YouTube player helper

If a JSON exists at `media-info/{video_id}.json` with `{ "watch_url": "..." }`, the app renders a player that:
- Seeks to the matched second.
- Displays **current time** and computed **frame** via `FPS` read from `map-keyframes/{video_id}.csv`.
- Allows seeking by **frame number**.

---

## 🛠️ Troubleshooting

- **No results / missing data**  
  - Check `DATA_ROOT` and ensure all necessary folders exist.  
  - Verify `clip-features-32/{video_id}.npy` and `map-keyframes/{video_id}.csv` are present.
- **BM25 errors**  
  - Confirm `text_bm25.db` exists for the selected batch and `docs` is **FTS5**.  
  - Ensure the `text` column is at index `2` for `snippet(docs, 2, ...)` or adjust the column index.
- **Gemini errors**  
  - Provide `GOOGLE_API_KEY`.  
  - Install both `google-genai` and `google-generativeai`, or unify the imports in code.
- **Images not found**  
  - The app tries fallbacks (±N frames, canonical lists, or scans the folder). Check zero‑padding width per batch (B1 uses 3–4, B2 often 4).

---

## 🧪 Minimal smoke test

1) Run with **BM25 Keyword Search** and query `"công an"` to validate FTS.  
2) Switch to **CLIP Multi‑aspects Search**; enter a short VI description (no Gemini).  
3) Enter an API key and click **Enhance Query** to verify Gemini and rerank.

---

## 📜 License

This README documents how to run the provided app. Use at your own risk. Dataset licenses and video links must follow their original terms.
