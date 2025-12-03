def load_raw_text_data(path: str) -> str:
    """Đọc toàn bộ nội dung file văn bản thô từ đường dẫn cho trước."""
    with open(path, encoding='utf-8') as f:
        return f.read()
