# Lab 5 Part 4 - Report

## 1. Mục tiêu
- Áp dụng pipeline RNN cho bài toán NER trên bộ CoNLL 2003 theo đúng yêu cầu lab.
- Luyện tập cách tải dữ liệu bằng Hugging Face `datasets`, xây dựng từ điển và lớp `Dataset` tùy chỉnh.
- Huấn luyện, đánh giá và phân tích định tính một mô hình BiLSTM đơn giản; chuẩn bị cơ sở để so sánh với xu hướng Transformer.

## 2. Dữ liệu và tiền xử lý
- Dùng `load_dataset("conll2003")` (phiên bản script cũ nên cần `datasets==2.19.1`). Các split có 14 041 câu train, 3 250 câu validation và 3 453 câu test.
- Nhãn theo chuẩn IOB: B/I-{PER, ORG, LOC, MISC} và `O`. Sau khi chuyển toàn bộ các mã số sang string, thêm nhãn đặc biệt `<PAD>` để phục vụ loss.
- Từ điển: tạo `word_to_ix` từ tập train, bổ sung `<PAD>` và `<UNK>`; không lower-case để giữ tín hiệu viết hoa quan trọng cho NER.
- `NERDataset` mã hóa từng câu thành tensor chỉ số từ/nhãn; `collate_fn` dùng `pad_sequence` để đệm batch và trả về chiều dài thật cho việc mask.

## 3. Kiến trúc mô hình
- Embedding 128 chiều (padding idx trỏ tới `<PAD>`).
- BiLSTM 256 hidden/unit mỗi chiều (512 đặc trưng tổng cộng) nhằm khai thác bối cảnh hai chiều; không thêm CRF để giữ cấu trúc gần với lab POS.
- Head tuyến tính chiếu 512 chiều sang số lượng nhãn.
- Hàm `predict_sentence` tách câu theo whitespace, ánh xạ `<UNK>` và suy ra nhãn qua `argmax`.

## 4. Cấu hình huấn luyện
- Batch size 64, optimizer Adam (lr = 1e-3), 5 epoch trên CPU.
- Loss: `nn.CrossEntropyLoss(ignore_index=pad_tag_idx)` để bỏ qua tokeen đệm.
- Theo dõi train/val accuracy mỗi epoch, lưu checkpoint tốt nhất tại `Lab5_part4/models/ner_bilstm.pt`.

## 5. Kết quả định lượng

| Split | Accuracy |
| --- | --- |
| Train | 0.9949 |
| Validation | 0.9492 |
| Test | 0.9322 |

- Đường cong trong `results/ner_bilstm_report.json` cho thấy loss giảm đều (0.63 → 0.05) và khoảng cách train–val duy trì <5%, chứng tỏ mô hình chưa bị overfit nặng dù không có regularization.
- Sai số trên test chủ yếu đến từ thực thể MISC/ORG dài hoặc có từ viết tắt nằm ngoài vocab.

## 6. Phân tích định tính
- **Thành công:** câu "VNU University is located in Hanoi" được gán đúng `B-ORG/I-ORG` cho cụm "VNU University" và `B-LOC` cho "Hanoi", cho thấy BiLSTM tận dụng được tín hiệu viết hoa + cấu trúc "in <Location>".
- **Sai lệch nhãn tổ chức:** trong ví dụ "CRICKET - <UNK> TAKE OVER AT TOP ..." (xem `ner_bilstm_examples.json`), token `<UNK>` thực chất là tên đội bóng nhưng bị ánh xạ UNK nên model gán `O`. Điều này lặp lại với các chữ viết tắt hiếm -> cần subword/char embedding.
- **Chưa ổn định với thực thể dài:** câu kể về "Phil Simmons" và các CLB (Leicestershire, Somerset) cho thấy model gán `B-MISC` cho "West"/"Indian" nhưng bỏ lỡ `I-PER` của "Phil Simmons" và đánh nhầm "Somerset" thành `O`. Việc không dùng CRF khiến mô hình khó duy trì ràng buộc IOB cho chuỗi dài.
- **Xử lý phụ thuộc xa:** câu liệt kê nhiều CLB (Essex, Derbyshire, Surrey, Kent) vẫn được gán đúng phần lớn, chứng tỏ BiLSTM truyền được thông tin dài hạn khi thực thể nằm cách nhau vài chục token.

## 7. Ví dụ dự đoán
```
Câu: VNU University is located in Hanoi
Dự đoán: [(VNU, B-ORG), (University, I-ORG), (is, O), (located, O), (in, O), (Hanoi, B-LOC)]
```

## 8. Nhận xét
- BiLSTM thuần đạt ~0.95 val / ~0.93 test accuracy: tốt cho baseline nhưng vẫn thấp hơn các mô hình Transformer + CRF (~0.97 F1 trên CoNLL).
- Hạn chế chính: không có wordpiece/character embedding nên thất bại với tên viết tắt hoặc token `<UNK>`; chưa áp dụng CRF để đảm bảo chuỗi IOB hợp lệ.
