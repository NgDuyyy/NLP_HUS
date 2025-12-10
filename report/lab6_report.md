# Lab 6 – Part 1 - Report
 
**Môi trường:** Python venv tại `Lab1_2/.venv`, `transformers==4.57.3`, chạy CPU với `torch`.

## 1. Tóm tắt mục tiêu
- Ôn lại các khái niệm cốt lõi của Transformer (encoder, decoder, self-attention) và ba họ omô hình chính.
- Thực hành các API `pipeline` của Hugging Face cho ba bài toán NLP kinh điển.
- Lưu lại mã chạy được và kết quả (`src/lab6/transformer_demos.py`, `result/lab6/lab6_part1_outputs.json`) để tái sử dụng.

## 2. Bài thực hành
Các thí nghiệm được chạy trong `transformer_demos.py`. Mỗi phần dưới đây trình bày cấu hình, kết quả quan sát và phần trả lời câu hỏi.

### 2.1 Masked Language Modeling (Encoder-only)
- **Pipeline:** `pipeline("fill-mask", model="bert-base-uncased")`
- **Câu đầu vào:** `Hanoi is the [MASK] of Vietnam.`
- **Top 5 dự đoán:**
  1. `capital` (0.9991)
  2. `center` (6.66e-05)
  3. `birthplace` (6.12e-05)
  4. `headquarters` (5.24e-05)
  5. `city` (5.21e-05)

**C1.** *Mô hình có chọn "capital"?*  
Có. `capital` đứng đầu với xác suất ~99.9%, vì vậy nên trường `contains_capital` trong JSON bằng `true`.

**C2.** *Vì sao BERT (encoder-only) phù hợp?*  
BERT quan sát hai chiều và được huấn luyện bằng Masked Language Modeling, nên khi dự đoán `[MASK]` nó đồng thời dùng cả ngữ cảnh trái và phải để suy ra từ bị ẩn. Decoder-only không nhìn được tương lai nên kém chính xác hơn cho bài toán này.

### 2.2 Next Token Prediction (Decoder-only)
- **Pipeline:** `pipeline("text-generation", model="gpt2", max_length=50, do_sample=False)`
- **Prompt:** `The best thing about learning NLP is`
- **Đoạn sinh mẫu:**

```
The best thing about learning NLP is that it's easy to learn. It's not hard to learn. ...
```
(Greedy decoding không lấy mẫu khiến GPT-2 lặp lại cụm “It's not hard to learn”.)

**C1.** *Kết quả có hợp lý?*  
Ở mức tương đối. Nội dung vẫn liên quan đến học NLP nhưng nhanh chóng lặp lại, phản ánh giới hạn của greedy decoding và quy mô nhỏ của GPT-2.

**C2.** *Vì sao GPT phù hợp?*  
GPT là mô hình tự hồi quy: tối đa hóa xác suất token kế tiếp dựa trên chuỗi đã sinh. Mục tiêu huấn luyện này trùng khớp với bài toán sinh văn bản, nên kiến trúc decoder-only phát huy tốt khi tiếp tục prompt.

### 2.3 Sentence Representation (Mean Pooling)
- **Model/tokenizer:** `AutoModel.from_pretrained("bert-base-uncased")`
- **Câu:** `This is a sample sentence.`
- **Kỹ thuật:** Mean pooling trên `last_hidden_state` và dùng `attention_mask` để loại padding.
- **Kết quả:** Vector lưu trong JSON (các phần tử đầu: `[-0.0639, -0.4284, ...]`).

**C1.** *Kích thước vector và tham số liên quan?*  
`embedding_dim = 768`, đúng bằng `hidden_size` của BERT-base. Mọi vector token của mô hình đều có 768 chiều nên câu sau khi pooling cũng giữ kích thước này.

**C2.** *Tại sao cần attention_mask khi Mean Pooling?*  
Padding chỉ là token giả để cân bằng độ dài. Nếu không mask, các giá trị này sẽ kéo trung bình về 0 và làm sai biểu diễn. Nhân với mặt nạ giúp chỉ các token thực đóng góp vào tổng và mẫu số.

## 3. Dependency Parsing Practice (spaCy)

Phần này hiện thực toàn bộ yêu cầu trong `lab6_dependency_parsing_pandoc.pdf`. Mã nguồn nằm tại `src/lab6/dependency_parsing.py` và sinh kết quả dưới `result/lab6/lab6_dependency_parsing_outputs.json`.

### 3.1 Hàm `find_main_verb`
- Input: chuỗi "The cat chased the mouse and the dog watched them.".
- Chiến lược: duyệt toàn bộ token, ưu tiên token có `dep_ == ROOT` và `pos_` thuộc {VERB, AUX}; nếu không có, lấy bất kỳ ROOT nào để tránh trả về `None`.
- Kết quả JSON: `main_verb = "chased"`, đúng với gốc của mệnh đề đầu tiên, đồng thời ghi lại nhãn `dependency = ROOT` để thuận tiện kiểm tra.

### 3.2 Gộp cụm danh từ thủ công
- Không dùng `Doc.noun_chunks` mà duyệt từng noun/proper-noun, gom các modifier (`amod`, `compound`, `det`, `poss`, `nummod`, `quantmod`) đứng trước danh từ để tạo Span.
- Ví dụ câu "The big, fluffy white cat is sleeping on the warm mat." cho ra hai cụm chính: `The big, fluffy white cat` và `the warm mat`, được nối vào JSON với chỉ số `start/end` để đối chiếu trong `doc`.

### 3.3 Đường đi tới ROOT
- Hàm `get_path_to_root(token)` lặp theo `token.head` cho đến khi gặp ROOT hoặc self-loop (phòng hờ spaCy edge cases), đồng thời lưu từng cặp `(text, dep, pos)` giúp debug.
- Với token "startup" trong câu "Apple is looking at buying U.K. startup for $1 billion.", đường đi ghi nhận lần lượt `startup -> buying -> at -> looking`, minh họa cấu trúc phụ thuộc "dobj → pcomp → prep → ROOT".

### 3.4 Lưu vết chạy
- Script tự tạo thư mục `result/lab6/` nếu thiếu, sau đó dump JSON bằng UTF-8. Thao tác này giữ dữ liệu thô trong repo nhưng nằm ngoài `data/` để tránh va chạm với file lớn.
- Khi lần đầu chạy, chương trình cố tải `en_core_web_md`; nếu thiếu sẽ gọi `spacy.cli.download`. Trong trường hợp không thể tải, fallback về `en_core_web_sm` và thông báo rõ ràng.


