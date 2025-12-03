# Lab 4 Report: Word Embeddings with Word2Vec

## 1. Các bước thực hiện
1. Chuẩn bị môi trường Python và mở notebook `lab4_word_embeddings.ipynb`.
2. Khởi tạo lớp `WordEmbedder` để tải mô hình pre-trained (`glove-wiki-gigaword-50`) và xây dựng các hàm tiện ích (vector từ, similarity, most similar, document embedding).
3. Thực hiện các bài tập chính với mô hình pre-trained: lấy vector của `king`, tính similarity giữa các cặp từ, và liệt kê các từ gần nghĩa của `computer`.
4. Áp dụng hàm `embed_document` để biểu diễn câu "The queen rules the country." thành vector bằng trung bình các word vector hợp lệ.
5. Thực hiện phần Bonus: đọc dữ liệu Universal Dependencies (UD English-EWT), huấn luyện Word2Vec với gensim, lưu mô hình và kiểm thử kết quả `most_similar`.
6. Thực hiện phần Advanced: chạy hàm `run_spark_word2vec` với PySpark trên tập C4 mẫu để huấn luyện Word2Vec phân tán, thống kê tần suất token và truy xuất synonym.

## 2. Hướng dẫn chạy code
- **Colab**
  1. Mở notebook `lab4_word_embeddings.ipynb` trên Google Colab.
  2. Bỏ comment dòng `!pip install gensim pyspark tqdm requests` ở Cell 3 để cài thư viện.
  3. Chạy tuần tự từng cell. Với phần Bonus, đảm bảo tải `Lab3_part2/data/UD_English-EWT/en_ewt-ud-train.txt` lên Colab hoặc chỉnh lại đường dẫn.
  4. Với phần Advanced, tải file `data/c4-train.00000-of-01024-30K.json.gz` lên cùng thư mục làm việc và chỉnh `json_path` nếu cần.
- **Local**
  1. Tạo venv và `pip install -r requirements.txt` (gensim, pyspark, tqdm, requests).
  2. Mở notebook bằng Jupyter Lab/Notebook hoặc VS Code.
  3. Đảm bảo các đường dẫn dữ liệu giống cấu trúc repo, sau đó chạy lần lượt các cell.

## 3. Phân tích kết quả
- **Mô hình pre-trained (GloVe 50d)**
  - Similarity: `sim(king, queen) ≈ 0.73` lớn hơn `sim(king, man) ≈ 0.53`, phản ánh quan hệ ngữ nghĩa gần gũi hơn giữa `king` và `queen`.
  - `most_similar('computer')` trả về danh sách gồm `computers`, `software`, `pc`, `hardware`, `workstation`, ... chứng tỏ mô hình học được ngữ cảnh công nghệ.
  - Document embedding của câu mẫu có norm khác 0, xác nhận ít nhất một token nằm trong vocabulary.
- **Mô hình tự huấn luyện (gensim Word2Vec trên UD English-EWT)**
  - Dataset ~12k câu sau tiền xử lý; mô hình Skip-gram 100 chiều.
  - `most_similar('computer')` trả về các token liên quan như `software`, `desktop`, `terminal` nhưng mức độ ổn định phụ thuộc vào kích thước tập và tham số `min_count`.
  - So với pre-trained, mô hình tự train cho kết quả hạn chế hơn do dữ liệu nhỏ và thiên về hội thoại/viết báo.
- **Mô hình Spark Word2Vec (C4 subset)**
  - Token tần suất cao chủ yếu là từ chức năng; nhiều token công nghệ hiếm nên dễ bị loại nếu `min_count` lớn.
  - Fallback hiển thị synonym của một token ngẫu nhiên trong vocabulary giúp xác nhận mô hình đã học.

## 4. Khó khăn và giải pháp
- **Đa định dạng dữ liệu UD**: File có thể ở dạng CoNLL-U hoặc plain text. Hàm `load_ud_sentences` đã được viết để tự phát hiện định dạng và vẫn trả về danh sách câu hợp lệ (,txt)
- **Save mô hình Word2Vec**: Gensim yêu cầu đường dẫn dạng string; đã dùng `str(model_path)` để tránh lỗi khi dùng `Path`.

