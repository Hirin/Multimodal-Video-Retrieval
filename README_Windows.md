# HCMC AI Competition - Windows Setup Guide

## Cấu hình môi trường cho Windows

### 1. Cấu hình biến môi trường

Tạo file `.env` (hoặc copy từ `env_config.txt`) trong thư mục gốc với nội dung:

```env
# HCMC AI Competition Configuration
# Windows path for data root (use forward slashes for cross-platform compatibility)
DATA_ROOT=E:/HCMC-AI/

# Google API Key for Gemini (optional)
GOOGLE_API_KEY=your_api_key_here
```

### 2. Cấu trúc thư mục dữ liệu

Đảm bảo thư mục `E:/HCMC-AI/` có cấu trúc như sau:

```
E:/HCMC-AI/
├── b1/
│   ├── clip-features-32/
│   ├── map-keyframes/
│   ├── keyframes/
│   ├── video/
│   ├── clip_keep_idxs/
│   ├── text_bm25.db
│   └── media-info/
└── b2/
    ├── clip-features-32/
    ├── map-keyframes/
    ├── keyframes/
    ├── video/
    ├── clip_keep_idxs/
    ├── text_bm25.db
    └── media-info/
```

### 3. Tính năng chính

Ứng dụng hỗ trợ **2 chế độ tìm kiếm** với **bộ lọc nâng cao**:

#### 🔧 **Search Filters** (Mới!)
- **Dataset Selection**: Chọn "Both Batches", "Batch 1", hoặc "Batch 2"
- **❌ Exclude Series**: Loại trừ series (vd: `L01,L02,K01`)
- **❌ Exclude Videos**: Loại trừ video cụ thể (vd: `L01V001,L01V002`)
- **Real-time feedback** về số lượng filters đang active
- **Áp dụng cho cả BM25 và CLIP search**

**Ví dụ sử dụng Filters:**
- Exclude series: `L01,L02` - Loại trừ tất cả videos từ L01 và L02
- Exclude videos: `L01V001,L01V002` - Loại trừ 2 videos cụ thể
- Combine: Dataset "Batch 1" + Exclude `L01` = Chỉ tìm trong L02+

#### 🔍 **BM25 Keyword Search** (Mới!)
- Tìm kiếm trực tiếp trong transcript của video
- Hỗ trợ cú pháp SQLite FTS5: `AND`, `OR`, `NOT`, phrase search
- **Mapping chính xác** từ timestamp sang keyframe sử dụng map-keyframes CSV
- **Hiển thị keyframe đã lọc duplicate** sử dụng parquet data (keyframes_keepers, manifest_dedup)
- **Full text với highlight tự động** - hiển thị toàn bộ text với từ khóa được highlight
- **Timestamp và keyframe number** chính xác từ parquet data
- **Keyframe đã lọc duplicate** - chỉ hiển thị keyframes canonical

**Ví dụ BM25 queries:**
- `cát AND điêu AND khắc` - Tìm video có cả 3 từ
- `cát OR "điêu khắc"` - Tìm video có "cát" hoặc cụm "điêu khắc"
- `cát NOT biển` - Tìm video có "cát" nhưng không có "biển"

**Cách hoạt động:**
- **Dataset & Filters**: Áp dụng dataset selection và exclude filters trước khi search
- BM25 tìm kiếm trong transcript và trả về timestamp + full text chính xác
- Ứng dụng đọc file `map-keyframes/{video_id}.csv` để mapping timestamp → keyframe number
- **Lọc duplicate**: Sử dụng `keyframes_keepers.parquet` (B1) và `manifest_dedup.parquet` (B2)
- Chỉ hiển thị keyframes canonical (không phải duplicate)
- **Full text được highlight** tự động với `<b>` tags cho từ khóa
- **Filters được áp dụng lại** cho kết quả cuối cùng
- Ví dụ: timestamp 220.12s → keyframe 0146.jpg (canonical only)

**Tùy chỉnh tìm kiếm:**
- 🔧 **"Search Filters"** - Cấu hình dataset và exclude filters
- 🎯 **"Max aspects"** - Giới hạn số aspects Gemini tạo
- 📚 **"Query Syntax Help"** - Hướng dẫn cú pháp BM25
- 🤖 **Gemini Model** - Chọn giữa Pro hoặc Flash model

#### 🎨 **CLIP Visual Search** (Mặc định)
- Tìm kiếm theo mô tả hình ảnh
- Sử dụng AI để hiểu ý nghĩa của query
- Tìm kiếm dựa trên nội dung video
- **Filters được áp dụng** giống như BM25 search

**Ví dụ CLIP queries:**
- `điêu khắc cát` - Tìm videos về điêu khắc cát
- `trượt ván trên cát` - Tìm videos về trượt ván trên cát
- `lễ hội điêu khắc` - Tìm videos về lễ hội điêu khắc

### 4. Chạy ứng dụng

#### Cách 1: Sử dụng Batch Script (khuyến nghị)
```bash
run_streamlit.bat
```

#### Cách 2: Chạy trực tiếp
```bash
cd E:\HCMC-AI-Competition
python -m streamlit run streamlit_app.py --server.port 8501 --server.headless true
```

### 5. Các thay đổi đã thực hiện để hỗ trợ Cross-Platform

1. **Biến môi trường**: Đã thay đổi từ `HCMC_DATA_ROOT` thành `DATA_ROOT`
2. **Path động**: Thay thế tất cả path hardcoded bằng path dựa trên `DATA_ROOT`
3. **Pathlib**: Sử dụng `pathlib.Path` để xử lý path cross-platform
4. **Forward slashes**: Sử dụng forward slashes trong path để tương thích với cả Windows và Linux

### 6. Lưu ý quan trọng

- **Path format**: Luôn sử dụng forward slashes (/) trong file .env, không sử dụng backslashes (\)
- **DATA_ROOT**: Phải kết thúc bằng dấu /
- **File encoding**: Đảm bảo file .env được lưu với encoding UTF-8 (không BOM)

### 6. Troubleshooting

Nếu gặp lỗi path:
1. Kiểm tra file `.env` có tồn tại và có nội dung chính xác
2. Đảm bảo đường dẫn trong `DATA_ROOT` chính xác và thư mục tồn tại
3. Thử chạy với Python để kiểm tra biến môi trường:
   ```python
   import os
   print(os.environ.get('DATA_ROOT'))
   ```

### 7. Migration từ Linux sang Windows

Nếu bạn di chuyển từ Linux sang Windows:

1. Thay đổi tất cả path từ `/mnt/shared/HCMC-AI/` thành `E:/HCMC-AI/`
2. Cập nhật file `.env` với path Windows mới
3. Đảm bảo sử dụng forward slashes thay vì backslashes
4. Kiểm tra lại cấu trúc thư mục dữ liệu

### 8. Gemini API Configuration (Tùy chọn)

#### 🔑 **Thiết lập API Key**
1. Truy cập [Google AI Studio](https://aistudio.google.com)
2. Tạo API key mới
3. Copy API key vào file `.env`:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

#### 🎯 **Cấu hình Model & Parameters**
- **Model Selection**:
  - `gemini-2.5-pro`: Model chất lượng cao, phản hồi chi tiết
  - `gemini-2.5-flash`: Model tốc độ cao, phản hồi nhanh
- **Max Aspects**: Số lượng aspects Gemini tạo (để trống = tự động, nhập số = giới hạn)
- **Temperature**: Độ sáng tạo (0.0-2.0)
- **Top-P**: Độ đa dạng (0.0-1.0)

#### 🧪 **Test Connection**
- Sử dụng button "🧪 Test Gemini API" để kiểm tra kết nối
- Nếu không có API key, hệ thống sẽ dùng fallback translation

#### 💡 **Tips sử dụng Gemini**
- **Pro**: Dùng cho query phức tạp, cần độ chính xác cao
- **Flash**: Dùng cho query đơn giản, cần tốc độ nhanh
- **Max Aspects**: Tăng cho query đa dạng, giảm cho query tập trung
- **Temperature**: Giảm cho câu trả lời nhất quán, tăng cho sáng tạo

### 9. Các file được tạo/cập nhật

- `streamlit_app.py`: Cập nhật để sử dụng `DATA_ROOT` và path động
- `run_streamlit.bat`: Script chạy cho Windows
- `env_config.txt`: Template cho file .env
- `README_Windows.md`: Hướng dẫn này
