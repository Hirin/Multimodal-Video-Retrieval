# HCMC AI 2025 – CLIP + BM25 Search (Streamlit App)

Ứng dụng truy xuất nhanh, đơn giản cho bộ dữ liệu video AIC25 / HCMC AI 2025.  
Kết hợp **tìm kiếm cửa sổ đa khía cạnh CLIP** với **BM25 transcript/OCR** và tùy chọn **bộ mở rộng truy vấn Gemini**.

---

## ✨ Chức năng của ứng dụng

- **Tìm kiếm CLIP Đa khía cạnh**: Mã hóa nhiều prompt khía cạnh và tìm **cửa sổ thời gian** tốt nhất cho mỗi video bằng cách sử dụng sliding max-pool theo khía cạnh, sau đó tổng hợp với trọng số.
- **Sắp xếp lại Transcript ("Ultra-Window")**: Đối với các ứng viên hàng đầu, mở cửa sổ chặt `[t0-Δ, t1+Δ]` và đếm **FTS5 hits** cho **unigrams/cụm từ** tiếng Việt được suy ra từ truy vấn + seeds, sau đó kết hợp `clip_n` và `text_score`.
- **Tìm kiếm từ khóa BM25 (độc lập)**: Truy vấn transcript/OCR trực tiếp bằng cú pháp FTS5, với highlighting cụm từ và preview keyframe.
- **Bộ mở rộng truy vấn Gemini (tùy chọn)**: Chuyển đổi prompt ngôn ngữ tự nhiên thành **khía cạnh (EN prompts + trọng số)** và **seeds tiếng Việt** cho bước sắp xếp lại. Fallback sang dịch offline nếu không có API key.
- **YouTube Player Overlay**: Nhảy đến vị trí khớp và hiển thị ánh xạ **thời gian → frame** trực tiếp bằng FPS từ metadata CSV.

---

## 📦 Cấu trúc thư mục & hợp đồng dữ liệu

Đặt `DATA_ROOT` thành thư mục chứa hai batch `b1/` và `b2/` với cấu trúc sau (các trường tối thiểu được hiển thị):

```
DATA_ROOT/
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
- **Dữ liệu**: Biến `DATA_ROOT` nên trỏ đến thư mục chứa `b1/` và `b2/` như mô tả bên dưới (tối thiểu: `clip-features-32/*.npy`, `map-keyframes/*.csv`, `text_bm25.db`).

## 🔧 Cài đặt

> Khuyến nghị Python 3.10+. GPU là tùy chọn (chỉ mã hóa CLIP text).

```bash
# Core
pip install streamlit pandas numpy pillow deep-translator

# CLIP text encoder (hoặc open-clip hoặc CLIP gốc)
pip install open-clip-torch          # ưu tiên
# Tùy chọn thay thế:
# pip install git+https://github.com/openai/CLIP.git

# Gemini (SDK chính thức mới được sử dụng trong toàn bộ ứng dụng)
pip install google-genai>=0.8.0

# Parquet (tùy chọn nhưng khuyến nghị cho metadata nhanh)
pip install pyarrow fastparquet
```

> Ứng dụng hiện chỉ sử dụng SDK **mới** `google-genai` (`google.genai.Client`).

---

## ⚙️ Cấu hình

Bạn có thể sử dụng file `.env` tại root của dự án (ứng dụng đọc nó **không cần deps dotenv bên ngoài**):

```env
DATA_ROOT=/mnt/shared/HCMC-AI/
GOOGLE_API_KEY=your_gemini_key_here
```

Tất cả env vars cũng có thể được export bình thường.

---

## ▶️ Chạy

Lưu mã của bạn dưới dạng `app.py` (hoặc bất kỳ tên nào) và chạy:

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
  - `BM25 Keyword Search` (tìm kiếm text độc lập)

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
- `text_score = 0.5*any_n + 0.5*phr_n` (có thể điều chỉnh)  
- **Điểm kết hợp**: `fused = 0.6*clip_n + 0.4*text_score` (có thể điều chỉnh)

> Các nút điều chỉnh được expose trong mã: `w_clip`, `w_text`, `text_weights = {"any": a, "phr": b}`, `Δ=DELTA_SEC`.

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

---

## 🎯 Tính năng chú thích video

Ứng dụng hiện bao gồm khả năng chú thích video toàn diện cho các tác vụ thu thập và gắn nhãn dữ liệu:

### 📝 Các loại tác vụ

- **KIS (Hệ thống nhận dạng Keyframe)**: Thu thập chú thích frame đơn
- **QA (Trả lời câu hỏi)**: Thu thập chú thích frame với văn bản đáp án
- **TRAKE (Theo dõi)**: Thu thập kết hợp frame đa scene

### 🎮 Trình phát video tương tác

- **Thanh trượt Range Dual-thumb**: Chọn khoảng frame với phản hồi trực quan
- **Thanh trượt Frame**: Điều hướng qua các frame video
- **Go to Frame**: Nhảy đến số frame cụ thể
- **Hiển thị thời gian**: Hiển thị thời gian hiện tại ở định dạng `mm:ss:ff`
- **Tự động bắt đầu**: Video bắt đầu tại timestamp được chỉ định

### 📊 Giao diện thu thập dữ liệu

- **Nhập số câu hỏi**: Chỉ định số lượng câu hỏi để chú thích
- **Lựa chọn tác vụ**: Chọn giữa các tác vụ KIS, QA, hoặc TRAKE
- **Nhập đáp án**: Cho tác vụ QA, nhập văn bản đáp án
- **Quản lý Scene**: Cho tác vụ TRAKE, quản lý nhiều scene với input động

### 🎨 Tính năng UI nâng cao

- **Nút Scene**: Lựa chọn scene trực quan với trạng thái mã màu (đỏ=trống, xanh=đã điền)
- **Tự động tiến**: Tự động chuyển sang scene tiếp theo sau khi thêm dữ liệu
- **Nút Current**: Đặt ranh giới range thành frame hiện tại
- **Điều khiển Step**: Tăng/giảm giá trị range
- **Xác thực thời gian thực**: Xác thực input với thông báo lỗi

### 💾 Quản lý dữ liệu

- **Preview Result**: Xem trước trực tiếp dữ liệu đã thu thập (giới hạn: 100 entries)
- **Delete Range**: Xóa entries bằng cách chỉ định khoảng hàng
- **Xuất CSV**: Lưu chú thích với định dạng `query-<số_câu>-<task>.csv`
- **Xử lý lỗi**: Xác thực toàn diện và thông báo lỗi
- **Phản hồi thành công**: Thông báo xác nhận cho tất cả hoạt động

### 🔧 Tính năng kỹ thuật

- **Định dạng khoảng cách**: Định dạng CSV đúng với khoảng cách sau dấu phẩy
- **Xác thực tác vụ**: Đảm bảo số câu hỏi và lựa chọn tác vụ trước khi lưu
- **Xác thực file**: Ngăn chặn lưu file trống hoặc không hợp lệ
- **Hệ thống thông báo**: Thông báo fade-in/fade-out cho phản hồi người dùng
- **Layout phản hồi**: Tối ưu cho các kích thước màn hình khác nhau

### 📋 Quy trình sử dụng

1. **Thiết lập**: Nhập số câu hỏi và chọn loại tác vụ
2. **Điều hướng**: Sử dụng thanh trượt và điều khiển để tìm frame mong muốn
3. **Thu thập**: Thêm frame đơn hoặc khoảng bằng nút Add/Add Batch
4. **Quản lý**: Xem lại, chỉnh sửa hoặc xóa dữ liệu đã thu thập trong Preview Result
5. **Xuất**: Lưu chú thích cuối cùng dưới dạng file CSV

---

## 🛠️ Khắc phục sự cố

- **Không có kết quả / thiếu dữ liệu**  
  - Kiểm tra `DATA_ROOT` và đảm bảo tất cả thư mục cần thiết tồn tại.  
  - Xác minh `clip-features-32/{video_id}.npy` và `map-keyframes/{video_id}.csv` có mặt.
- **Lỗi BM25**  
  - Xác nhận `text_bm25.db` tồn tại cho batch đã chọn và `docs` là **FTS5**.  
  - Đảm bảo cột `text` ở chỉ số `2` cho `snippet(docs, 2, ...)` hoặc điều chỉnh chỉ số cột.
- **Lỗi Gemini**  
  - Cung cấp `GOOGLE_API_KEY`.  
  - Cài đặt cả `google-genai` và `google-generativeai`, hoặc thống nhất imports trong mã.
- **Không tìm thấy hình ảnh**  
  - Ứng dụng thử fallbacks (±N frames, danh sách chuẩn, hoặc quét thư mục). Kiểm tra độ rộng zero-padding mỗi batch (B1 sử dụng 3–4, B2 thường 4).

---

## 🧪 Kiểm tra khói tối thiểu

1) Chạy với **BM25 Keyword Search** và truy vấn `"công an"` để xác thực FTS.  
2) Chuyển sang **CLIP Multi‑aspects Search**; nhập mô tả VI ngắn (không Gemini).  
3) Nhập API key và click **Enhance Query** để xác minh Gemini và sắp xếp lại.

---

## 📜 Giấy phép

README này tài liệu hóa cách chạy ứng dụng được cung cấp. Sử dụng với rủi ro của bạn. Giấy phép dataset và liên kết video phải tuân theo điều khoản gốc của chúng.
