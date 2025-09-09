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

### 3. Chạy ứng dụng

#### Cách 1: Sử dụng Batch Script (khuyến nghị)
```bash
run_streamlit.bat
```

#### Cách 2: Chạy trực tiếp
```bash
cd E:\HCMC-AI-Competition
python -m streamlit run streamlit_app.py --server.port 8501 --server.headless true
```

### 4. Các thay đổi đã thực hiện để hỗ trợ Cross-Platform

1. **Biến môi trường**: Đã thay đổi từ `HCMC_DATA_ROOT` thành `DATA_ROOT`
2. **Path động**: Thay thế tất cả path hardcoded bằng path dựa trên `DATA_ROOT`
3. **Pathlib**: Sử dụng `pathlib.Path` để xử lý path cross-platform
4. **Forward slashes**: Sử dụng forward slashes trong path để tương thích với cả Windows và Linux

### 5. Lưu ý quan trọng

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

### 8. Các file được tạo/cập nhật

- `streamlit_app.py`: Cập nhật để sử dụng `DATA_ROOT` và path động
- `run_streamlit.bat`: Script chạy cho Windows
- `env_config.txt`: Template cho file .env
- `README_Windows.md`: Hướng dẫn này
