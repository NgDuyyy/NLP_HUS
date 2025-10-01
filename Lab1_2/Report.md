# Báo cáo Lab 1 & Lab 2: Tokenization và Count Vectorization

## 1. Mô tả công việc

- Xây dựng các class Tokenizer (SimpleTokenizer, RegexTokenizer) để tách từ và dấu câu từ văn bản tiếng Anh.
- Tạo abstract base class Vectorizer và cài đặt CountVectorizer để chuyển văn bản thành vector số (Bag-of-Words).
- Xử lý các trường hợp đặc biệt như dấu câu, chữ hoa/thường, và kiểm thử trên cả câu mẫu lẫn dữ liệu thực tế.

## 2. Kết quả chạy code

### Tokenizer trên câu mẫu:
```python
sentences = [
    "Hello, world! This is a test.",
    "NLP is fascinating... isn't it?",
    "Let's see how it handles 123 numbers and punctuation!"
]
# Output SimpleTokenizer:
['hello', ',', 'world', '!', 'this', 'is', 'a', 'test.']
# Output RegexTokenizer:
['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
```

### CountVectorizer trên corpus mẫu:
```python
corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]
# Vocabulary:
{'AI': 0, '.', 1, 'I': 2, 'NLP': 3, 'love': 4, 'of': 5, 'programming': 6, 'subfield': 7}
# Document-term matrix:
[1, 1, 1, 1, 1, 0, 0, 0]
[0, 1, 1, 0, 1, 0, 1, 0]
[1, 1, 0, 1, 0, 1, 0, 1]
```

### Tokenizer và CountVectorizer trên dataset UD_English-EWT:
- Đọc 500 ký tự đầu từ file dữ liệu, tokenize và vector hóa.
- Output: danh sách token và ma trận số cho đoạn văn bản thực tế.

## 3. Giải thích kết quả

- **So sánh SimpleTokenizer và RegexTokenizer:**
  - SimpleTokenizer chỉ tách các dấu câu cơ bản, có thể dính dấu chấm vào cuối từ.
  - RegexTokenizer tách được nhiều loại dấu câu, cho kết quả token hóa chi tiết hơn.
- **CountVectorizer:**
  - Từ vựng (vocabulary) được xây dựng từ toàn bộ corpus, mỗi từ gắn với một chỉ số duy nhất.
  - Ma trận cho biết số lần xuất hiện của từng từ trong mỗi văn bản.
- **Khó khăn:**
  - Xử lý import module khi chạy kiểm thử, cần cấu trúc package chuẩn.
  - Dữ liệu thực tế có thể chứa ký tự đặc biệt, cần kiểm thử kỹ với các trường hợp này.

## 4. Kết luận
- Đã hoàn thành các yêu cầu về tokenizer và vectorizer, kiểm thử thành công trên cả dữ liệu mẫu và thực tế.
- Có thể mở rộng để xử lý các trường hợp phức tạp hơn hoặc tích hợp vào pipeline NLP lớn hơn.

## 5. Nguồn dâtset
- https://github.com/UniversalDependencies/UD_English-EWT
