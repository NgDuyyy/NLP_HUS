# Kho lưu trữ Môn học Xử lý Ngôn ngữ Tự nhiên & Ứng dụng (NLP & Apps)

Repo tổng hợp toàn bộ bài tập, notebook, dữ liệu mẫu và báo cáo của học phần NLP & Applications tại HUS. Mục tiêu là duy trì một cấu trúc thống nhất để giảng viên dễ chấm và sinh viên dễ tiếp tục phát triển.

## Lời mở đầu

- Mỗi lab đều có báo cáo chi tiết bằng Markdown, được liệt kê trong `report/README.md`.
- Dữ liệu dung lượng lớn được giữ ngoài Git; chỉ giữ script và mô tả schema để đảm bảo repo nhẹ và dễ clone.
- Hệ thống mã nguồn gồm cả Python (pip/venv) và Scala (sbt) nên có thể tái sử dụng cho nhiều bài toán khác nhau.

## Cấu trúc dự án

```
./
├── data/        # Thư mục dữ liệu chuẩn hóa theo từng lab (chỉ giữ mô tả/schema)
├── notebook/    # Notebook Jupyter cho các bài thực hành
├── report/      # Báo cáo Lab 1 → 6, more_research_part1 và tài liệu tham khảo
├── src/         # Mã nguồn chính (Python & Scala) chia theo lab: lab1_2 → lab6
├── test/        # Bộ kiểm thử hợp nhất (pytest/Scala tests)
├── sbt/         # Cấu hình build dành cho các project Scala
├── requirements.txt
└── README.md
```

## Bắt đầu

1. **Clone repo**
	 ```powershell
	 git clone https://github.com/NgDuyyy/NLP_HUS.git
	 cd NLP_HUS
	 ```
2. **Thiết lập Python venv**
	 ```powershell
	 python -m venv .venv
	 .\.venv\Scripts\activate
	 pip install -r requirements.txt
	 ```
3. **Chuẩn bị môi trường Scala/Spark (nếu chạy các lab dùng sbt)**
	 - Cài JDK 17+ và sbt 1.9+
	 - Cập nhật biến môi trường `JAVA_HOME` nếu cần.

## Hướng dẫn chạy

- **Lab Python** (lab1_2, lab3, lab4, lab5, lab6)
	1. Kích hoạt venv và cài dependency theo `requirements.txt` hoặc `src/labX/requirements.txt` (nếu có).
	2. Chạy thử nghiệm: `pytest test/labX` hoặc script tương ứng trong `src/labX/`.
	3. Notebook liên quan nằm trong `notebook/` (ví dụ: `notebook/lab5_pytorch_introduction.ipynb`).

- **Lab Scala/Spark** (lab2, lab2_alt)
	1. Di chuyển tới thư mục `src/lab2/` hoặc `src/lab2_alt/`.
	2. Chạy `sbt run` hoặc `sbt test` tùy bài.
	3. Kết quả/log sẽ xuất hiện dưới `data/lab2/...`.

- **Báo cáo & nghiên cứu**
	- Lưu trữ tại `report/`. Xem bảng tổng hợp và liên kết đầy đủ trong `report/README.md`.
	- Phần mở rộng nghiên cứu Text-to-Speech nằm trong `report/more_research_part1.md`.

## Tác giả

- **Git:** `NgDuyyy` (little_cat)
- **Môn học:** NLP & Applications
- **Trường:** Hanoi University of Science – VNU
