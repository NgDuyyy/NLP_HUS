# MÔ TẢ CẤU TRÚC THƯ MỤC DỮ LIỆU

Thư mục chính chứa dữ liệu là `data/`. Cấu trúc này cho thấy repository đang chứa ít nhất hai bộ dữ liệu chính: `hwu` và `UD_English-EWT`, cùng với các file dữ liệu tổng hợp khác.

## Thư mục `data/`

Thư mục này là nơi chứa các nguồn dữ liệu thô (raw data) và dữ liệu đã được tiền xử lý, phục vụ cho các bài Lab về NLP và các ứng dụng liên quan.

---

### Bộ dữ liệu `hwu` (HWU - Leeds University)

Bộ dữ liệu này thường được sử dụng cho các bài toán Phân loại Ý định (Intent Classification) hoặc các tác vụ liên quan đến Hội thoại/Trợ lý ảo.

| Tên File | Mô tả | Ứng dụng/Mục đích |
| :--- | :--- | :--- |
| `categories.json` | Chứa ánh xạ hoặc định nghĩa các lớp (categories) ý định. | Định nghĩa tập nhãn cho bài toán Phân loại ý định. |
| `test.csv` | Dữ liệu kiểm thử. | Đánh giá hiệu năng cuối cùng của mô hình. |
| `train.csv` | Dữ liệu huấn luyện chính. | Xây dựng và tối ưu hóa mô hình. |
| `train_5.csv` | Mẫu nhỏ (5%?) dữ liệu huấn luyện. | Huấn luyện thử nghiệm nhanh, kiểm tra lỗi (debugging). |
| `train_10.csv` | Mẫu nhỏ (10%?) dữ liệu huấn luyện. | Huấn luyện thử nghiệm nhanh, kiểm tra lỗi (debugging). |
| `val.csv` | Dữ liệu validation (thẩm định). | Theo dõi overfitting và điều chỉnh siêu tham số (hyperparameters). |

---

### Bộ dữ liệu `UD_English-EWT` (Universal Dependencies - English Web Treebank)

Bộ dữ liệu tiêu chuẩn này được sử dụng rộng rãi trong các tác vụ **Phân tích Cú pháp (Parsing)** và **Gán nhãn Chuỗi** (Sequence Labeling), thường là POS Tagging (Gán nhãn Từ loại) hoặc Dependency Parsing.

| Tên File | Mô tả | Định dạng | Ứng dụng/Mục đích |
| :--- | :--- | :--- | :--- |
| `en_ewt-ud-dev.conllu` | Tập dữ liệu Phát triển (Development). | CoNLL-U | Dùng cho Validation (kiểm thử trong quá trình phát triển). |
| `en_ewt-ud-train.conllu` | Tập dữ liệu Huấn luyện. | CoNLL-U | Dữ liệu chính để huấn luyện mô hình. |

[cite_start]***Lưu ý:** Bộ dữ liệu `UD_English-EWT` này được sử dụng trong **Lab 1: Text Tokenization (Task 3)** để kiểm thử tokenizer trên dữ liệu thực tế.* [cite: 36, 39]

---

### Các File và Dữ liệu khác trong `data/`

| Tên File | Mô tả | Loại |
| :--- | :--- | :--- |
| `c4-train.00000-of-01024.json.gz` | Có khả năng là một phần của bộ dữ liệu C4 (Colossal Clean Crawled Corpus) nén. | Dữ liệu ngôn ngữ lớn |
| `hwu.tar.gz` | File nén chứa toàn bộ thư mục `hwu` (dữ liệu thô). | Lưu trữ dữ liệu |
| `sentiments.csv` | Dữ liệu liên quan đến bài toán Phân tích Cảm xúc (Sentiment Analysis). | Dữ liệu phân loại |

---