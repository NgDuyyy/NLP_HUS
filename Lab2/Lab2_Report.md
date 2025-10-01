# Lab 02: Pipeline Xử Lý Dữ Liệu NLP với Apache Spark - Báo Cáo Thực Hiện

## 1. Các Bước Thực Hiện

### 1.1 Thiết Lập Môi Trường
Dự án được phát triển sử dụng công nghệ sau:
- Apache Spark 4.0.1 với Scala 2.13
- sbt 1.11.6 để quản lý build
- Java 19 runtime environment

### 1.2 Kiến Trúc Data Pipeline
Pipeline xử lý NLP được thực hiện theo các bước tuần tự sau:

**Bước 1: Tải Dữ Liệu**
- Đọc dữ liệu JSON nén từ file `c4-train.00000-of-01024-30K.json.gz`
- Sử dụng JSON reader tích hợp của Spark với tự động suy luận schema
- Tải 1000 bản ghi văn bản vào Spark DataFrame

**Bước 2: Cấu Hình Pipeline Tiền Xử Lý Văn Bản**
- Cấu hình RegexTokenizer để tách từ cấp độ từ sử dụng pattern `\\W+`
- Thiết lập StopWordsRemover để lọc bỏ các từ dừng tiếng Anh phổ biến
- Kích hoạt xử lý không phân biệt hoa thường và chuyển đổi về chữ thường

**Bước 3: Thiết Lập Vector Hóa Đặc Trưng**
- Cấu hình HashingTF với 20000 chiều đặc trưng để tính toán tần suất từ
- Thiết lập IDF (Inverse Document Frequency) transformer cho TF-IDF weighting
- Liên kết các thành phần thành một ML Pipeline thống nhất

**Bước 4: Thực Thi Pipeline**
- Lắp ráp tất cả thành phần thành một Spark ML Pipeline 4 giai đoạn
- Fit pipeline trên dữ liệu training để học IDF weights
- Transform dữ liệu qua tất cả các giai đoạn pipeline

**Bước 5: Lưu Trữ Kết Quả**
- Trích xuất feature vectors và văn bản gốc từ DataFrame đã xử lý
- Lưu kết quả vào `results/lab17_pipeline_output.txt` với định dạng tùy chỉnh
- Tạo log thực thi chi tiết trong thư mục `log/`

## 2. Thực Thi Code và Kết Quả

### 2.1 Tóm Tắt Thực Thi
Pipeline thực thi thành công với các chỉ số hiệu suất sau:
- Tổng thời gisn thực thi: 15 giây
- Thời gian tải dữ liệu: 2,58 giây
- Thời gian fit pipeline: 1,49 giây
- Thời gian transform dữ liệu: 0,68 giây
- Thời gian lưu kết quả: 0,23 giây

### 2.2 Thống Kê Xử Lý
- Tổng số tài liệu được xử lý: 1000
- Kích thước từ vựng duy nhất: 874 từ
- Trung bình tokens mỗi tài liệu: 18,10
- Trung bình filtered tokens mỗi tài liệu: 12,00
- Số chiều feature vector: 20000
- Độ thưa thớt vector: 99,93%

### 2.3 Định Dạng Output
Kết quả được lưu trong định dạng văn bản có cấu trúc với mỗi bản ghi chứa:
```
Text: [Văn bản tài liệu gốc]
Features: [Biểu diễn sparse vector với indices và values]
--------------------------------------------------------------------------------
```

## 3. Phân Tích Kết Quả

### 3.1 Hiệu Quả Tiền Xử Lý
Pipeline tiền xử lý văn bản đã thành công giảm từ vựng từ raw tokens thành các thuật ngữ có ý nghĩa:
- Loại bỏ stop words đã loại trừ khoảng 33% tokens (18,10 → 12,00 trung bình)
- RegexTokenizer xử lý hiệu quả dấu câu và ký tự đặc biệt
- Chuẩn hóa case đảm bảo biểu diễn token nhất quán

### 3.2 Chất Lượng Vector Hóa
Vector hóa TF-IDF tạo ra biểu diễn thưa thớt chất lượng cao:
- 99,93% độ thưa thớt cho thấy sử dụng bộ nhớ hiệu quả
- Không gian đặc trưng 20000 chiều cung cấp khả năng biểu diễn đầy đủ
- IDF weighting thành công giảm trọng số các từ phổ biến trong corpus

### 3.3 Hiệu Suất Pipeline
Spark ML Pipeline thể hiện đặc tính khả năng mở rộng tốt:
- Xử lý song song sử dụng tất cả 12 CPU cores có sẵn
- Sử dụng bộ nhớ hiệu quả với lazy evaluation
- Tính modular của pipeline cho phép thay thế thành phần dễ dàng

## 4. Khó Khăn Gặp Phải và Giải Pháp

### 4.1 Vấn Đề Tương Thích Windows-Hadoop
**Vấn đề**: Lưu kết quả bằng built-in writers của Spark thất bại do thiếu Hadoop utilities trên Windows.
**Lỗi**: `HADOOP_HOME and hadoop.home.dir are unset`
**Giải pháp**: Triển khai file I/O thủ công sử dụng Java PrintWriter, bỏ qua dependencies của Hadoop filesystem.

### 4.2 Vấn Đề Serialization Data Type
**Vấn đề**: Vector UDT (User Defined Type) của Spark không thể được serialized trực tiếp sang định dạng text.
**Lỗi**: `UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE`
**Giải pháp**: Trích xuất dữ liệu vector sử dụng `.collect()` và format thủ công features thành text có thể đọc được với indices và values.

### 4.3 Xung Đột Build System
**Vấn đề**: Các tham chiếu class bị bỏ lại gây lỗi compilation sau khi tái cấu trúc code.
**Lỗi**: Missing class references trong build artifacts
**Giải pháp**: Xóa các file có vấn đề và làm sạch build cache để đảm bảo trạng thái compilation nhất quán.

### 4.4 Tối Ưu Hóa Quản Lý Bộ Nhớ
**Vấn đề**: Feature vectors lớn gây vấn đề bộ nhớ trong quá trình collection.
**Giải pháp**: Triển khai selective data collection và streaming write operations để giảm thiểu memory footprint.

## 5. Chi Tiết Triển Khai Kỹ Thuật

### 5.1 Các Thành Phần Cốt Lõi
Triển khai bao gồm hai Scala objects chính:
- `Lab17_NLPPipeline.scala`: Triển khai tập trung vào bài tập 1, 2, 3 và 4
- `Lab17_DataPipeline.scala`: Data Pipeline

### 5.2 Kiến Trúc Pipeline
```
Raw Text Data → RegexTokenizer → StopWordsRemover → HashingTF → IDF → Feature Vectors
```

### 5.3 Chiến Lược Xử Lý Lỗi
Triển khai xử lý lỗi sử dụng pattern Try/Success/Failure của Scala:
- Khôi phục lỗi graceful với logging lỗi chi tiết
- Dọn dẹp tài nguyên trong finally blocks
- Thông báo lỗi thông tin với stack traces

## 6. Dependencies Và Thư Viện Bên Ngoài

### 6.1 Apache Spark Libraries
- `spark-core_2.13:4.0.1`: Chức năng Spark cốt lõi
- `spark-sql_2.13:4.0.1`: DataFrame và SQL operations
- `spark-mllib_2.13:4.0.1`: Các thành phần machine learning pipeline

### 6.2 Scala Standard Libraries
- `scala.util.{Try, Success, Failure}`: Xử lý lỗi
- `java.io.{PrintWriter, BufferedWriter, FileWriter}`: File I/O operations
- `java.time.LocalDateTime`: Tạo timestamp

### 6.3 Cấu Hình Build
Dự án sử dụng sbt với các dependencies chính sau:
```scala
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "4.0.1",
  "org.apache.spark" %% "spark-sql" % "4.0.1",
  "org.apache.spark" %% "spark-mllib" % "4.0.1"
)
```

## 7. Kết Luận
Pipeline xử lý 1000 tài liệu hiệu quả, tạo ra các biểu diễn đặc trưng có ý nghĩa phù hợp.
Tất cả các tiêu chí đánh giá đã được thỏa mãn.