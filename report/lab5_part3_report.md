# Lab 5 Phần 3

## 1. Mục tiêu
- Xây dựng mô hình gán nhãn từ loại (POS) dựa trên mạng nơ-ron tái hồi hai chiều cho bộ dữ liệu UD English-EWT.
- So sánh hành vi train/dev/test để xác định mô hình đơn giản này đang làm tốt hay thất bại ở đâu.
- Cung cấp đủ chỉ số định lượng, phân tích định tính và ví dụ minh họa theo yêu cầu trong pdf

## 2. Dữ liệu và tiền xử lý
- Nguồn dữ liệu: UD English-EWT r1.0 (`data/UD_English-EWT-r1.0/en-ud-{train,dev,test}.conllu`).
- Cách đọc file: bỏ qua dòng comment, bỏ những dòng biểu diễn từ ghép (id dạng `1-2`), chỉ giữ cặp `(FORM, UPOS)`(cột 2 và 4).
- Tạo từ vựng: chuyển chữ thường toàn bộ (`LOWERCASE=True`), xây `word_to_ix` từ tập train và thêm sẵn `<PAD>`, `<UNK>`; từ vụng nhãn gồm `<PAD>` và toàn bộ UPOS.
- Dataset/Dataloader: `POSDataset` mã hóa từng câu thành tensor; `pad_sequence` giúp đệm batch và trả về chiều dài thực để phân tích.

## 3. Kiến trúc mô hình
- Embedding: `nn.Embedding` kích thước 128, sử dụng `padding_idx` để giữ vector `<PAD>` bằng 0.
- Encoder: `nn.RNN` hai chiều (batch-first) với 256 hidden/unit cho mỗi chiều → 512 đặc trưng hợp nhất.
- Đầu phân loại: linear χếu 512 đặc trưng về số lượng nhãn, tạo logits ở từng thời điểm.
- Dự đoán: lấy argmax theo trục nhãn; hàm mất mát bỏ qua các vị trí đệm nên gradient chỉ truyền qua token thật.

## 4. Cấu hình huấn luyện
- Tối ưu: Adam, learning rate 1e-3; loss: CrossEntropyLoss(ignore_index=`pad_tag`).
- Batch size 64, huấn luyện 8 epoch trên CPU (máy cá nhân không có CUDA).
- Cố định seed 42 cho Python/NumPy/Torch để tái lập.
- Sau mỗi epoch: đo train/dev accuracy và lưu checkpoint tốt nhất tại `Lab5_part3/models/pos_rnn.pt`.

## 5. Kết quả định lượng

| Tập | Accuracy |
| --- | --- |
| Train | 0.9792 |
| Dev | 0.8968 |
| Test | 0.8881 |

- Xu hướng học (xem `results/pos_rnn_report.json`): dev accuracy tăng từ 0.78 (epoch 1) lên 0.887 (epoch 8) trong khi train loss giảm 0.92 → 0.11, cho thấy hội tụ ổn định với khoảng chênh overfit ~9%.
- Hidden size 256 cân bằng giữa năng lực biểu diễn và độ ổn định; bản RNN nông hơn đã bị dao động dev nên bị loại.

## 6. Phân tích định tính
- Điểm mạnh: Câu "president bush on tuesday nominated two individuals to replace retiring <UNK> on federal courts in the washington area ." được gán đúng toàn bộ; mô hình xử lý được chuỗi danh từ riêng dài và cấu trúc động từ nguyên mẫu.
- Yếu điểm: Với "from the ap comes this story :" mô hình gán `NOUN` cho "ap" thay vì `PROPN`, phản ánh khó khăn khi mất thông tin viết hoa do lowercase toàn bộ.
- Lỗi do token hiếm: Câu "bush nominated jennifer m. anderson ..." có nhiều chữ cái viết tắt trở thành `<UNK>`, dẫn đến dự đoán `NOUN` chung chung thay vì `PROPN`; cần thêm embedding theo ký tự hoặc subword để phân biệt chính xác.
- Thành công với phụ thuộc xa: "bush also nominated a. noel <UNK> <UNK> for a 15 - year term as associate judge of the district of columbia court of appeals , replacing john montague <UNK> ." gồm nhiều mệnh đề xen giữa nhưng mô hình vẫn giữ đúng cặp danh từ ghép "associate judge" và cụm giới từ dài phía sau nhờ RNN hai chiều truyền thông tin bối cảnh.
- Thất bại phụ thuộc xa: "bush nominated jennifer m. anderson for a 15 - year term as associate judge of the superior court of the district of columbia , replacing <UNK> w. <UNK> ." yêu cầu phân biệt các danh từ riêng nằm xa; khi các chữ cái đơn bị ẩn thành `<UNK>`, mô hình dự đoán `NOUN` cho "anderson" và `SCONJ` cho "as" nhưng lại trượt ở cụm "superior court" vì tín hiệu nhãn đúng nằm sau 10+ token. Đây là dấu hiệu RNN vanilla suy giảm ngữ cảnh dài và cần cơ chế gating (GRU/LSTM) để giữ thông tin tốt hơn.

## 7. Ví dụ dự đoán

```
Câu: I love NLP
Dự đoán: [(I, PRON), (love, VERB), (NLP, ADP)]
```

- Mô hình xử lý tốt đại từ và động từ nhưng gán "NLP" thành `ADP` do tập train hiếm gặp chữ viết tắt này và lowercase làm mất gợi ý viết hoa; hiện tượng khớp với phân tích định tính ở trên.
