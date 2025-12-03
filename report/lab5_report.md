# Lab5 — Part 2: Model comparison report


## 1. Bảng so sánh định lượng 


| Model | Validation F1 | Test F1 | Validation Acc | Test Acc |
|---|---:|---:|---:|---:|
| TF-IDF + LogisticRegression | 0.8645 | 0.8264 | 0.8663 | 0.8282 |
| Word2Vec(mean) + LogisticRegression | 0.6016 | 0.5517 | 0.6091 | 0.5552 |
| LSTM + pretrained emb (Model A) | 0.6455 | 0.6378 | 0.6546 | 0.6472 |
| LSTM + train-from-scratch emb (Model B) | 0.7688 | 0.7261 | 0.7744 | 0.7289 |


**Ghi chú về loss:** Không có los chung được lưu cho các baseline; loss huấn luyện cho LSTM được in trong quá trình huấn luyện (xem logs) và đã giảm ổn định qua các epoch. báo cáo trên dựa vào F1/Accuracy trên validation/test.


## 2. Phân tích định tính, ví dụ điển hình


Dưới đây là một số ví dụ từ tập test LSTM (Model B) dự đoán đúng trong khi TF-IDF/Word2Vec trung bình sai.


### Ví dụ (index 24)
- Text: please delete the wednesday evening alarm
- True intent id: 1
- TF-IDF pred: 38, Word2Vec pred: 1, LSTM pred: 1
**Phân tích:**
- Lý do LSTM có thể đúng: LSTM giữ thông tin thứ tự và ngữ cảnh, vì vậy những từ khóa phân tán hoặc phụ thuộc ngữ cảnh xa được mô hình chuỗi nắm bắt tốt hơn.
- Lý do TF-IDF/Word2Vec thất bại: TF-IDF chỉ xử lý tần suất/bi-gram và bỏ vị trí, Word2Vec trung bình làm mất thông tin vị trí và câu trúc.


### Ví dụ (index 30)
- Text: please turn off the alarm
- True intent id: 1
- TF-IDF pred: 31, Word2Vec pred: 31, LSTM pred: 1
**Phân tích:**
- Lý do LSTM có thể đúng: LSTM giữ thông tin thứ tự và ngữ cảnh, vì vậy những từ khóa phân tán hoặc phụ thuộc ngữ cảnh xa được mô hình chuỗi nắm bắt tốt hơn.
- Lý do TF-IDF/Word2Vec thất bại: TF-IDF chỉ xử lý tần suất/bi-gram và bỏ vị trí, Word2Vec trung bình làm mất thông tin vị trí và câu trúc.


### Ví dụ (index 42)
- Text: wake up time
- True intent id: 2
- TF-IDF pred: 11, Word2Vec pred: 33, LSTM pred: 2
**Phân tích:**
- Lý do LSTM có thể đúng: LSTM giữ thông tin thứ tự và ngữ cảnh, vì vậy những từ khóa phân tán hoặc phụ thuộc ngữ cảnh xa được mô hình chuỗi nắm bắt tốt hơn.
- Lý do TF-IDF/Word2Vec thất bại: TF-IDF chỉ xử lý tần suất/bi-gram và bỏ vị tríWord2Vec trung bình làm mất thông tin vị trí và câu trúc.


### Ví dụ (index 45)
- Text: please set an alarm clock for my next meeting with the team at three pm nẽt friday
- True intent id: 2
- TF-IDF pred: 8, Word2Vec pred: 8, LSTM pred: 2
**Phân tích:**
- Lý do LSTM có thể đúng: LSTM giữ thông tin thứ tự và ngữ cảnh, vì vậy những từ khóa phân tán hoặc phụ thuộc ngữ cảnh xa được mô hình chuỗi nắm bắt tốt hơn.
- Lý do TF-IDF/Word2Vec thất bại: TF-IDF chỉ xử lý tần suất/bi-gram và bỏ vị trí; Word2Vec trung bình làm mất thông tin vị trí và câu trúc.


### Ví dụ (index 64)
- Text: turn off media volume
- True intent id: 4
- TF-IDF pred: 31, Word2Vec pred: 34, LSTM pred: 4
**Phân tích:**
- Lý do LSTM có thể đúng: LSTM giữ thông tin thứ tự và ngữ cảnh, vì vậy những từ khóa phân tán hoặc phụ thuộc ngữ cảnh xa được mô hình chuỗi nắm bắt tốt hơn.
- Lý do TF-IDF/Word2Vec thất bại: TF-IDF chỉ xử lý tần suất/bi-gram và bỏ vị trí; Word2Vec trung bình làm mất thông tin vị trí và câu trúc.


## 3. Ví dụ LSTM sai nhưng TF-IDF đúng


### Ví dụ (index 20)
- Text: disable the alarm which is set at nine thirty pm
- True intent id: 1
- TF-IDF pred: 1, LSTM pred: 2
**Phân tích:**
- Lý do LSTM thất bại: có thể do dữ liệu ngắn, có từ khóa đặc trưng mạnh mà TF-IDF bắt được, LSTM có thể chưa đủ epoch hoặc model overfit/underfit.


### Ví dụ (index 36)
- Text: set a timer at five am seven days a week
- True intent id: 2
- TF-IDF pred: 2, LSTM pred: 8
**Phân tích:**
- Lý do LSTM thất bại: có thể do dữ liệu ngắn, có từ khóa đặc trưng mạnh mà TF-IDF bắt được, LSTM có thể chưa đủ epoch hoặc model overfit/underfit.


### Ví dụ (index 50)
- Text: could you please lower the tone
- True intent id: 3
- TF-IDF pred: 3, LSTM pred: 4
**Phân tích:**
- Lý do LSTM thất bại: có thể do dữ liệu ngắn, có từ khóa đặc trưng mạnh mà TF-IDF bắt được, LSTM có thể chưa đủ epoch hoặc model overfit/underfit.


### Ví dụ (index 52)
- Text: lower the speak volume
- True intent id: 3
- TF-IDF pred: 3, LSTM pred: 5
**Phân tích:**
- Lý do LSTM thất bại: có thể do dữ liệu ngắn, có từ khóa đặc trưng mạnh mà TF-IDF bắt được, LSTM có thể chưa đủ epoch hoặc model overfit/underfit.


### Ví dụ (index 56)
- Text: lower the volume to twenty
- True intent id: 3
- TF-IDF pred: 3, LSTM pred: 5
**Phân tích:**
- Lý do LSTM thất bại: có thể do dữ liệu ngắn, có từ khóa đặc trưng mạnh mà TF-IDF bắt được, LSTM có thể chưa đủ epoch hoặc model overfit/underfit.


## 4. Nhận xét chung về ưu và nhược điểm của từng phương pháp



- TF-IDF + LogisticRegression
  - Ưu: Nhanh, hiệu quả với dữ liệu có từ khóa đặc trưng, dễ triển khai.
  - Nhược: Không nắm bắt thứ tự từ, ngữ cảnh dài; kém với paraphrase nếu không có từ khóa.

- Word2Vec(mean) + LogisticRegression
  - Ưu: Mượt mà hơn TF-IDF về biểu diễn từ, có thể tóm tắt ý nghĩa từ.
  - Nhược: Trung bình vector mất thông tin vị trí và cấu trúc; kém khi câu có thông tin dựa trên trật tự.

- LSTM + Embedding (pretrained)
  - Ưu: Có kiến thức ngôn ngữ từ pretrained vectors; LSTM nắm được thứ tự và ngữ cảnh cục bộ.
  - Nhược: Nếu pretrained không khớp nhiệm vụ, cần fine-tune; chi phí tính toán cao.

- LSTM + Embedding (train-from-scratch)
  - Ưu: Embedding được tối ưu trực tiếp cho nhiệm vụ dẫn tới hiệu năng tốt hơn trong thí nghiệm này.
  - Nhược: Cần nhiều dữ liệu để học embedding tốt; dễ overfit nếu dữ liệu nhỏ.



## 5. Tài liệu tham khảo & tiêu chí báo cáo


- Tài liệu lab và các tiêu chí chấm trước đây đã được tuân thủ: trình bày kết quả định lượng, phân tích định tính, so sánh phương pháp và nêu ưu/nhược điểm.