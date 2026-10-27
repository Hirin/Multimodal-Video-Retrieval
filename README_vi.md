# Video Retrieval - CLIP + BM25 Search & Video Annotation (Streamlit App)

Ứng dụng truy xuất và chú thích video toàn diện cho bộ dữ liệu video AIC25.  
Kết hợp **tìm kiếm cửa sổ đa khía cạnh CLIP** với **BM25 transcript/OCR**, **bộ mở rộng truy vấn Gemini**, và **hệ thống chú thích video tương tác**.

### Phiên bản hiện tại và phiên bản cũ

- [`app.py`](app.py) là ứng dụng tìm kiếm đã tách thành các package `services/`, `utils/` và `components/`.
- [`app_original_backup.py`](app_original_backup.py) là phiên bản nguyên khối trước đó. File này vẫn chứa trình phát YouTube/local với thao tác chú thích KIS, QA, TRAKE và xuất CSV, cùng hàm `transcript_seed_bm25_search`. Các phần triển khai này chưa có trong ứng dụng đã tách module.
- Các tính năng trình phát/chú thích mô tả bên dưới thuộc phiên bản cũ. Ứng dụng module không import file này. Cần giữ lại cho đến khi các tính năng được chuyển sang và kiểm chứng; đây không phải bản sao dư thừa.

---

## ✨ Chức năng của ứng dụng

### 🔍 **Tìm kiếm Video**
- **Tìm kiếm CLIP Đa khía cạnh**: Mã hóa nhiều prompt khía cạnh và tìm **cửa sổ thời gian** tốt nhất cho mỗi video bằng cách sử dụng sliding max-pool theo khía cạnh, sau đó tổng hợp với trọng số.
- **Sắp xếp lại Transcript ("Ultra-Window")**: Đối với các ứng viên hàng đầu, mở cửa sổ chặt `[t0-Δ, t1+Δ]` và đếm **FTS5 hits** cho **unigrams/cụm từ** tiếng Việt được suy ra từ truy vấn + seeds, sau đó kết hợp `clip_n` và `text_score`.
- **Tìm kiếm từ khóa BM25 (độc lập)**: Truy vấn transcript/OCR trực tiếp bằng cú pháp FTS5, với highlighting cụm từ và preview keyframe.
- **Bộ mở rộng truy vấn Gemini (tùy chọn)**: Chuyển đổi prompt ngôn ngữ tự nhiên thành **khía cạnh (EN prompts + trọng số)** và **seeds tiếng Việt** cho bước sắp xếp lại. Fallback sang dịch offline nếu không có API key.
- **Tìm kiếm theo Video ID**: Tìm kiếm trực tiếp video cụ thể bằng ID.

### 🎥 **Trình phát Video**
- **YouTube Player**: Nhảy đến vị trí khớp và hiển thị ánh xạ **thời gian → frame** trực tiếp bằng FPS từ metadata CSV.
- **Kaggle Offline Player**: Tải video từ Kaggle về máy local, sau đó stream qua Flask server với khả năng điều hướng frame chính xác (không giới hạn dung lượng).
- **Dual-thumb Range Slider**: Chọn khoảng frame với phản hồi trực quan.
- **Go to Frame**: Nhảy đến số frame cụ thể.
- **Real-time Frame Display**: Hiển thị thời gian và frame hiện tại.

### 📝 **Hệ thống Chú thích Video**
- **KIS (Keyframe Identification System)**: Thu thập chú thích frame đơn
- **QA (Question Answering)**: Thu thập chú thích frame với văn bản đáp án
- **TRAKE (Tracking)**: Thu thập kết hợp frame đa scene với quản lý scene động
- **Batch Annotation**: Thu thập nhiều frame cùng lúc với range selection
- **Data Export**: Xuất chú thích dưới dạng CSV với định dạng chuẩn

---

## 📦 Cấu trúc thư mục & hợp đồng dữ liệu

Đặt biến môi trường `DATA_ROOT` trỏ đến thư mục chứa hai batch `b1/` và `b2/` với cấu trúc sau (các trường tối thiểu được hiển thị):

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
│  ├─ clip-features-32/         # npy mỗi video: (T, D) float32, CLIP features
│  │   ├─ L21_V001.npy
│  │   └─ ...
│  ├─ keyframes/                 # keyframes được trích xuất (hình ảnh)
│  │   └─ L21/V001/0001.jpg ...  # frame ids zero-padded (3–4 chữ số cho B1)
│  ├─ map-keyframes/             # bản đồ thời gian mỗi video (CSV)
│  │   └─ L21_V001.csv           # cột bắt buộc: pts_time (float), fps (float), n (int tùy chọn)
│  ├─ clip_keep_idxs/            # tùy chọn: chỉ số của frames chuẩn
│  │   └─ L21_V001_keep_idxs.npy
│  ├─ keyframes_keepers.parquet  # tùy chọn: bảng frames chuẩn cho B1
│  ├─ media-info/                # tùy chọn: JSON mỗi video với watch_url
│  │   └─ L21_V001.json          # { "watch_url": "https://youtu.be/..." }
│  └─ text_bm25.db               # cơ sở dữ liệu SQLite FTS5 (xem schema bên dưới)
└─ b2/
   ├─ clip-features-32/
   ├─ keyframes/                 # lưu ý: B2 thường sử dụng frame ids 4 chữ số
   ├─ map-keyframes/
   ├─ clip_keep_idxs/
   ├─ manifest_dedup.parquet     # tùy chọn: bao gồm cờ is_canonical
   ├─ media-info/
   └─ text_bm25.db
```

### SQLite FTS5 (BM25) schema

Mã mong đợi **bảng ảo FTS5** có tên `docs` bên trong `text_bm25.db` với ít nhất các cột sau:

- `video_id` (TEXT) — ví dụ: `L21_V001`
- `text` (TEXT) — văn bản transcript/OCR
- `t` (REAL) — timestamp (giây)
- các cột khác được phép

> Ứng dụng sử dụng:
>
> - `WHERE text MATCH ?` để tìm kiếm  
> - `snippet(docs, 2, '<b>', '</b>', '...', 50)` cho previews (chỉ số cột `2` → cột `text`)  
> - `bm25(docs)` để xếp hạng  
> - `SELECT ... WHERE video_id=? AND t BETWEEN ? AND ?` để đếm ultra-window

Đảm bảo bản build SQLite của bạn hỗ trợ **FTS5**.

### `map-keyframes/*.csv`

Các cột hữu ích mong đợi:
- `pts_time` (float) — giây (đơn điệu)
- `fps` (float) — tốc độ khung hình (hàng đầu tiên được sử dụng)
- `n` (int, tùy chọn) — số keyframe

Nếu `n` bị thiếu, ứng dụng ước tính frames từ thời gian và `fps` cộng với khoảng thời gian heuristic.

---

## 🧰 Yêu cầu tối thiểu

- **Python**: 3.10+ (khuyến nghị 3.11)
- **Hệ điều hành**: Windows 10/11, macOS, hoặc Linux
- **GPU**: Không bắt buộc (ứng dụng chỉ mã hóa CLIP text; CPU là đủ)
- **SQLite FTS5**: Bắt buộc cho BM25/FTS5; Python 3.10+ thường bao gồm FTS5. Nếu gặp lỗi FTS5, sử dụng phiên bản Python mới hơn.
- **Dữ liệu**: Biến môi trường `DATA_ROOT` nên trỏ đến thư mục chứa `b1/` và `b2/` như mô tả bên dưới (tối thiểu: `clip-features-32/*.npy`, `map-keyframes/*.csv`, `text_bm25.db`). Dataset có thể đặt ở bất kỳ đâu, miễn là khai báo đúng trong biến môi trường.

## 🔧 Cài đặt

> Khuyến nghị Python 3.10+. GPU là tùy chọn (chỉ mã hóa CLIP text).

### Tạo Virtual Environment (Khuyến nghị)
```bash
# Tạo virtual environment
python -m venv .venv

# Kích hoạt virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Kiểm tra đã kích hoạt thành công (sẽ thấy (.venv) ở đầu dòng)
```

### Cài đặt từ requirements.txt
```bash
pip install -r requirements.txt
```

### Cài đặt thủ công
```bash
# Core dependencies
pip install streamlit>=1.28.0 pandas>=1.3.0 numpy>=1.21.0 pillow>=8.0.0 deep-translator>=1.9.0

# CLIP text encoder
pip install open-clip-torch>=2.20.0

# Google Gemini API (SDK mới)
pip install google-genai>=0.8.0

# Additional utilities
pip install requests>=2.25.0 tqdm>=4.60.0 stopwordsiso>=0.1.0

# Kaggle Hub (for offline video player)
pip install kagglehub

# Optional: Parquet support for faster metadata loading
pip install pyarrow fastparquet
```

> **Lưu ý**: Ứng dụng sử dụng SDK **mới** `google-genai` (`google.genai.Client`) thay vì `google-generativeai`.

---

## ⚙️ Cấu hình

Bạn có thể sử dụng file `.env` tại root của dự án (ứng dụng đọc nó **không cần deps dotenv bên ngoài**):

```env
DATA_ROOT=/mnt/shared/HCMC-AI/
GOOGLE_API_KEY=your_gemini_key_here
```

Tất cả env vars cũng có thể được export bình thường.

### 🔑 **Kaggle API Setup**

Để sử dụng tính năng "Kaggle Offline" video player:

1. **Lấy Kaggle API Token**:
   - Đăng nhập vào [Kaggle](https://www.kaggle.com/)
   - Vào Account → API → Create New API Token
   - Tải file `kaggle.json` về máy

2. **Cấu hình Kaggle API**:
   ```bash
   # Tạo thư mục .kaggle trong cùng thư mục code
   mkdir .kaggle
   
   # Copy file kaggle.json vào thư mục .kaggle
   copy kaggle.json .kaggle\        # Windows
   cp kaggle.json .kaggle/         # Linux/macOS
   
   # Đặt quyền phù hợp (Linux/macOS)
   chmod 600 .kaggle/kaggle.json
   ```

3. **Cài đặt Kaggle Hub**:
   ```bash
   pip install kagglehub
   ```

4. **Test kết nối**:
   ```python
   import kagglehub
   # Test download một dataset nhỏ để kiểm tra API
   ```

### 📁 **Video Storage Management**

**Vị trí lưu video**:
- Video được tải về: `./videos/` (trong thư mục dự án)
- Format tên file: `{video_id}.mp4` (ví dụ: `K01_V001.mp4`)

**Tự động dọn dẹp**:
```bash
# Xóa tất cả video đã tải
rm -rf ./videos/*

# Xóa video cũ hơn 7 ngày (Linux/macOS)
find ./videos -name "*.mp4" -mtime +7 -delete

# Xóa video cũ hơn 7 ngày (Windows PowerShell)
Get-ChildItem ./videos -Filter "*.mp4" | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)} | Remove-Item
```

**Cấu hình dung lượng**:
- Video được tải từ Kaggle về máy local trước, sau đó stream qua Flask server
- Thư mục `videos/` có thể phát triển lớn theo thời gian
- Khuyến nghị: Dọn dẹp định kỳ hoặc tăng dung lượng ổ cứng

---

## ▶️ Chạy

### Kích hoạt Virtual Environment (nếu sử dụng)
```bash
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### Chạy ứng dụng
```bash
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
```

Mở URL local được in trong trình duyệt.

---

## 🖥️ Tổng quan giao diện

### 🔍 Giao diện tìm kiếm

- **Bộ lọc tìm kiếm**  
  - Dataset: `Both / Batch 1 / Batch 2`  
  - Loại trừ Series: `L01,L02,K01,...`  
  - Loại trừ Videos: `L01_V001,...`

- **Cấu hình Gemini API** (tùy chọn)  
  - Nhập API key, chọn model (`gemini-2.5-flash` hoặc `gemini-2.5-pro`)  
  - Thanh trượt Temperature / Top‑P  
  - **Test Gemini API** (kiểm tra nhanh)

- **Cài đặt**  
  - `Window Second (mm:ss)` → độ dài cửa sổ trượt CLIP (mặc định `00:40`)  
  - `Skip Intro (mm:ss)` → bắt đầu tính điểm sau thời gian này  
  - `Rerank Window (mm:ss)` → ±Δ giây xung quanh cửa sổ CLIP cho sắp xếp lại transcript (mặc định `00:20`)  
  - `Top K` → số lượng video để hiển thị

- **Chế độ tìm kiếm**
  - `CLIP Multi‑aspects Search` (không sắp xếp lại)
  - `CLIP Multi‑aspects Search + Rerank` (cần Gemini để xây dựng seeds tiếng Việt)
  - `Transcript Seed Search (BM25)` (tìm kiếm với seeds từ Gemini)
  - `BM25 Keyword Search` (tìm kiếm text độc lập)
  - `Search by Video ID` (tìm kiếm video cụ thể)

- **Hộp truy vấn**
  - Ngôn ngữ tự nhiên (VI/EN) cho chế độ CLIP
  - Chế độ BM25 chấp nhận truy vấn FTS5 (xem popover cú pháp trong ứng dụng)

### 🎯 Giao diện chú thích video

- **Cấu hình tác vụ**
  - Số câu hỏi: Số lượng câu hỏi để chú thích
  - Loại tác vụ: Lựa chọn KIS, QA, hoặc TRAKE
  - Nhập đáp án: Cho tác vụ QA (khi áp dụng)

- **Điều khiển trình phát video**
  - Thanh trượt Frame: Điều hướng qua các frame video
  - Thanh trượt Range: Chọn khoảng frame với dual thumbs
  - Go to Frame: Nhảy đến số frame cụ thể
  - Nút Current: Đặt range thành frame hiện tại
  - Điều khiển Step: Tăng/giảm giá trị

- **Thu thập dữ liệu**
  - Nút Add: Thu thập frame đơn
  - Nút Add Batch: Thu thập khoảng frame
  - Quản lý Scene: Cho tác vụ TRAKE (input scene động)
  - Auto-advance: Tự động chuyển sang scene tiếp theo

- **Quản lý dữ liệu**
  - Preview Result: Xem trước trực tiếp dữ liệu đã thu thập
  - Delete Range: Xóa entries theo số hàng
  - Save CSV: Xuất chú thích với định dạng đúng
  - Notifications: Hệ thống phản hồi thành công/lỗi

---

## 🔤 Cú pháp truy vấn BM25 (FTS5)

Ví dụ bạn có thể paste vào **BM25 Keyword Search**:

```sql
cát AND điêu AND khắc
cát OR "điêu khắc"
cát NOT biển
"cột đá" AND (10 OR 12)
"công an"
```

Phím tắt được ứng dụng nhận dạng:
- `&` → `AND`
- `|` → `OR`
- `-` → `NOT`

Ứng dụng highlight các từ/cụm từ khớp trong preview văn bản đầy đủ.

---

## 🧠 Cách tính điểm hoạt động

### 1) Cửa sổ đa khía cạnh CLIP
- Xây dựng nhiều **aspect prompts** → mã hóa qua CLIP text encoder → chuẩn hóa `L2`.
- Đối với mỗi video, tải **frame features** được tính trước `X` có hình dạng `(T, D)` và `times` từ `map-keyframes` (căn chỉnh qua `keep_idxs` nếu có).
- Đối với cửa sổ trượt `[i..e)`, tính **max similarity** mỗi khía cạnh trong cửa sổ và **tổng với trọng số**.
- Giữ cửa sổ tốt nhất `win = (t0, t1)` và **peak frames mỗi khía cạnh** để preview.

### 2) Sắp xếp lại transcript ultra-window
- Từ **truy vấn tiếng Việt** + **seeds**, xây dựng:
  - `unigrams` (≤28) và `phrases` (≤18)
  - `q_any = uni ∪ phrases` và `q_phr = phrases only`
- Đối với mỗi video ứng viên, đếm FTS5 hits trong `[t0-Δ, t1+Δ]`:
  - `hits_any = COUNT(q_any)`
  - `hits_phr = COUNT(q_phr)`
- Chuẩn hóa bằng min-max → `clip_n`, `any_n`, `phr_n`
- `text_score = 0.3*any_n + 0.7*phr_n` (có thể điều chỉnh)  
- **Điểm kết hợp**: `fused = 0.4*clip_n + 0.6*text_score` (có thể điều chỉnh)

> Các nút điều chỉnh được expose trong mã: `w_clip=0.4`, `w_text=0.6`, `text_weights = {"any": 0.3, "phr": 0.7}`, `Δ=DELTA_SEC`.

---

## 🤖 Bộ mở rộng Gemini (khía cạnh + seeds)

- Hướng dẫn hệ thống yêu cầu JSON:  
  - `aspects`: tên (VI), `prompts` (2–4 EN), `weight` ∈ **[0.8, 1.3]**  
  - `seeds`: `unigrams` / `phrases` tiếng Việt
- Nếu API thất bại hoặc không có key, **fallback** translator xây dựng khía cạnh tối thiểu từ bản dịch theo dòng và seeds trống.

> Trong **chế độ CLIP cơ bản**, bạn vẫn có thể **điều chỉnh trọng số khía cạnh thủ công** qua thanh trượt (0.5 → 1.6) để sắp xếp lại ranking mà không cần mã hóa lại.

---

## 🎬 Trình phát YouTube helper

Nếu JSON tồn tại tại `media-info/{video_id}.json` với `{ "watch_url": "..." }`, ứng dụng render trình phát:
- Tìm đến giây khớp.
- Hiển thị **thời gian hiện tại** và **frame** được tính qua `FPS` đọc từ `map-keyframes/{video_id}.csv`.
- Cho phép tìm kiếm theo **số frame**.

## 🎬 Kaggle Offline Player với Flask Streaming

Ứng dụng sử dụng Flask server để stream video từ local storage:
- **Download Process**: Video được tải từ Kaggle về `./videos/` trước
- **Flask Streaming**: Server Flask chạy trên port 8801 để stream video
- **No Size Limit**: Không bị giới hạn 200MB của Kaggle API
- **Frame Navigation**: Điều hướng frame chính xác với FPS từ metadata
- **Auto-start**: Video bắt đầu tại timestamp được chỉ định

---

## 🎯 Tính năng chú thích video

Ứng dụng bao gồm hệ thống chú thích video toàn diện với giao diện tương tác cao:

### 📝 **Các loại tác vụ**

- **KIS (Keyframe Identification System)**: Thu thập chú thích frame đơn
- **QA (Question Answering)**: Thu thập chú thích frame với văn bản đáp án
- **TRAKE (Tracking)**: Thu thập kết hợp frame đa scene với quản lý scene động

### 🎮 **Trình phát video tương tác**

- **Dual-thumb Range Slider**: Chọn khoảng frame với phản hồi trực quan
- **Frame Navigation**: Điều hướng qua các frame video với thanh trượt
- **Go to Frame**: Nhảy đến số frame cụ thể bằng input
- **Real-time Display**: Hiển thị thời gian và frame hiện tại
- **Auto-start**: Video bắt đầu tại timestamp được chỉ định

### 📊 **Giao diện thu thập dữ liệu**

- **Task Configuration**: Chọn loại tác vụ và số câu hỏi
- **QA Answer Input**: Nhập đáp án cho tác vụ QA
- **Scene Management**: Quản lý nhiều scene cho tác vụ TRAKE
- **Batch Collection**: Thu thập nhiều frame cùng lúc

### 🎨 **Tính năng UI nâng cao**

- **Visual Scene Selection**: Nút scene với trạng thái mã màu (đỏ=trống, xanh=đã điền)
- **Auto-advance**: Tự động chuyển sang scene tiếp theo
- **Current Frame Button**: Đặt ranh giới range thành frame hiện tại
- **Step Controls**: Tăng/giảm giá trị range với nút ▲/▼
- **Real-time Validation**: Xác thực input với thông báo lỗi

### 💾 **Quản lý dữ liệu**

- **Live Preview**: Xem trước trực tiếp dữ liệu đã thu thập (giới hạn: 100 entries)
- **Range Deletion**: Xóa entries bằng cách chỉ định khoảng hàng
- **CSV Export**: Lưu chú thích với định dạng `query-<số_câu>-<task>.csv`
- **Error Handling**: Xác thực toàn diện và thông báo lỗi
- **Success Notifications**: Thông báo xác nhận cho tất cả hoạt động

### 📋 **Quy trình sử dụng**

1. **Thiết lập**: Nhập số câu hỏi và chọn loại tác vụ
2. **Điều hướng**: Sử dụng thanh trượt và điều khiển để tìm frame mong muốn
3. **Thu thập**: Thêm frame đơn hoặc khoảng bằng nút Add/Add Batch
4. **Quản lý**: Xem lại, chỉnh sửa hoặc xóa dữ liệu đã thu thập trong Preview Result
5. **Xuất**: Lưu chú thích cuối cùng dưới dạng file CSV

---

## 🛠️ Khắc phục sự cố

### **Dữ liệu và Cấu hình**
- **Không có kết quả / thiếu dữ liệu**  
  - Kiểm tra `DATA_ROOT` và đảm bảo tất cả thư mục cần thiết tồn tại.  
  - Xác minh `clip-features-32/{video_id}.npy` và `map-keyframes/{video_id}.csv` có mặt.
  - Kiểm tra quyền truy cập file và đường dẫn có chứa ký tự đặc biệt.
- **Virtual Environment Issues**
  - Đảm bảo đã kích hoạt virtual environment: `source .venv/bin/activate` (Linux/macOS) hoặc `.venv\Scripts\activate` (Windows)
  - Kiểm tra Python path: `which python` (Linux/macOS) hoặc `where python` (Windows)
  - Nếu gặp lỗi import, thử cài đặt lại: `pip install -r requirements.txt`

### **Tìm kiếm BM25**
- **Lỗi BM25**  
  - Xác nhận `text_bm25.db` tồn tại cho batch đã chọn và `docs` là **FTS5**.  
  - Đảm bảo cột `text` ở chỉ số `2` cho `snippet(docs, 2, ...)` hoặc điều chỉnh chỉ số cột.
  - Kiểm tra SQLite version hỗ trợ FTS5: `sqlite3 --version`
  - Test query đơn giản: `"công an"` hoặc `cát AND điêu`

### **Gemini API**
- **Lỗi Gemini**  
  - Cung cấp `GOOGLE_API_KEY` hợp lệ từ https://aistudio.google.com
  - Sử dụng SDK mới: `google-genai>=0.8.0` (không phải `google-generativeai`)
  - Test API connection bằng nút "Test Gemini API" trong ứng dụng
  - Kiểm tra quota và rate limits của API

### **Video và Hình ảnh**
- **Không tìm thấy hình ảnh**  
  - Ứng dụng thử fallbacks (±N frames, danh sách chuẩn, hoặc quét thư mục)
  - Kiểm tra độ rộng zero-padding mỗi batch (B1 sử dụng 3–4, B2 thường 4) [[memory:8547778]]
  - Xác minh `keyframes/{video_id}/` có chứa file hình ảnh
- **Video không phát được**
  - YouTube: Kiểm tra URL hợp lệ và video không bị hạn chế
  - Kaggle Offline: Đảm bảo có quyền truy cập Kaggle dataset
  - Local server: Kiểm tra port 8801 không bị chiếm dụng
  - Flask streaming: Kiểm tra Flask server đang chạy trên port 8801
- **Kaggle API Issues**
  - Kiểm tra file `kaggle.json` trong `./.kaggle/` (trong thư mục dự án)
  - Test API bằng Python: `import kagglehub`
  - Đảm bảo đã accept dataset terms trên Kaggle website
  - Kiểm tra quota và rate limits của Kaggle API
  - Kiểm tra kết nối internet và firewall settings
- **Video Storage Issues**
  - Kiểm tra dung lượng ổ cứng còn trống
  - Xóa video cũ: `rm -rf ./videos/*` (Linux/macOS) hoặc `Remove-Item ./videos/*` (Windows)
  - Kiểm tra quyền ghi vào thư mục `./videos/`

### **Chú thích Video**
- **Lỗi chú thích**
  - Đảm bảo nhập số câu hỏi hợp lệ (> 0)
  - Cho tác vụ QA: nhập đáp án trước khi thêm frame
  - Cho tác vụ TRAKE: nhập số scene và frame cho mỗi scene
  - Kiểm tra giới hạn 100 entries trong preview

### **Performance**
- **Ứng dụng chạy chậm**
  - Giảm `Top K` để hiển thị ít video hơn
  - Tăng `Skip Intro` để bỏ qua phần đầu video
  - Sử dụng GPU nếu có cho CLIP encoding
  - Kiểm tra RAM và CPU usage

---

## 🧪 Kiểm tra khói tối thiểu

### **Test cơ bản**
1) **BM25 Keyword Search**: Truy vấn `"công an"` để xác thực FTS5
2) **CLIP Multi-aspects Search**: Nhập mô tả VI ngắn (không cần Gemini)
3) **Video ID Search**: Tìm `K01_V001` hoặc `L21_V001`

### **Test nâng cao**
4) **Gemini Integration**: Nhập API key và test "Enhance Query"
5) **Rerank**: Chạy "CLIP Multi-aspects Search + Rerank"
6) **Video Annotation**: Test chú thích với tác vụ KIS/QA/TRAKE

### **Test Video Player**
7) **YouTube Player**: Click "Watch Video" → "Youtube Online"
8) **Kaggle Offline**: Click "Watch Video" → "Kaggle Offline"

---

## 📚 Ví dụ sử dụng

### **Tìm kiếm cơ bản**
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

### **Cấu hình nâng cao**
```bash
# Environment variables
DATA_ROOT=/path/to/HCMC-AI/
GOOGLE_API_KEY=your_gemini_key_here

# Streamlit options
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
```

### **Annotation Workflow**
1. **Setup**: Chọn tác vụ (KIS/QA/TRAKE), nhập số câu hỏi
2. **Navigate**: Sử dụng frame slider để tìm vị trí mong muốn
3. **Collect**: Add single frame hoặc Add Batch range
4. **Review**: Xem Preview Result, xóa nếu cần
5. **Export**: Save CSV với format chuẩn

---

## 📜 Giấy phép

README này tài liệu hóa cách chạy ứng dụng được cung cấp. Sử dụng với rủi ro của bạn. Giấy phép dataset và liên kết video phải tuân theo điều khoản gốc của chúng.
