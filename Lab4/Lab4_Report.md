# Lab 4: Text Classification - Báo cáo và Phân tích

## 1. Các bước thực hiện

### Bước 1: Tạo TfidfVectorizer
- **File**: `src/models/tfidf_vectorizer.py`
- **Mục đích**: Chuyển đổi text thành TF-IDF features
- **Các bước**:
  1. Tokenize text sử dụng RegexTokenizer
  2. Xây dựng vocabulary từ training corpus
  3. Tính IDF cho mỗi từ: `IDF = log((N+1)/(df+1)) + 1`
  4. Transform documents thành TF-IDF vectors: `TF-IDF = (count/total_terms) * IDF`

### Bước 2: Implement TextClassifier
- **File**: `src/models/text_classifier.py`
- **Thành phần**:
  - `__init__()`: Khởi tạo với vectorizer instance
  - `fit()`: Train LogisticRegression model
    - Vectorize texts với `vectorizer.fit_transform()`
    - Initialize `LogisticRegression(solver='liblinear')`
    - Train với `model.fit(X, labels)`
  - `predict()`: Dự đoán labels cho new texts
    - Transform texts với `vectorizer.transform()`
    - Return `model.predict(X)`
  - `evaluate()`: Tính metrics
    - Accuracy, Precision, Recall, F1-score từ sklearn.metrics

### Bước 3: Create Test Cases
- **Basic Test** (`test/lab5_test.py`):
  - Dataset 12 samples
  - Train/test split 80/20
  - Train và evaluate model
  
- **Hugging Face Test** (`test/lab5_test_huggingface.py`):
  - Load dataset từ Hugging Face
  - 3 experiments với dataset sizes khác nhau
  - So sánh performance và overfitting

- **PySpark Test** (`test/lab5_spark_sentiment_analysis.py`):
  - Spark ML Pipeline với Tokenizer, StopWordsRemover, HashingTF, IDF
  - LogisticRegression classifier
  - Evaluation với Spark evaluators

---

## 2. Hướng dẫn chạy code

### Cài đặt Dependencies
```bash
# Navigate to Lab4 folder
cd Lab4

# Install required packages
pip install -r requirements.txt
```

**Dependencies cần thiết:**
- scikit-learn >= 1.0.0
- numpy >= 1.21.0
- datasets >= 2.0.0 (cho Hugging Face)
- pyspark >= 3.3.0 (optional, cho Spark test)

### Chạy Tests

#### Test 1: Basic Classification
```bash
python test/lab5_test.py
```
**Expected Output:**
- Training: 9 samples
- Testing: 3 samples
- Accuracy: ~67%
- Execution time: < 1 second

#### Test 2: Hugging Face Dataset (Recommended)
```bash
python test/lab5_test_huggingface.py
```
**Expected Output:**
- Download dataset từ Hugging Face
- 3 experiments: Small (100), Medium (500), Large (2000)
- Accuracy: 84%, 76%, 80% respectively
- Execution time: ~30-45 seconds

#### Test 3: PySpark (Advanced)
```bash
python test/lab5_spark_sentiment_analysis.py
```
**Expected Output:**
- Spark ML Pipeline execution
- Accuracy và F1-score
- Execution time: varies based on system

---

## 3. Thông tin Dataset

### Dataset: Twitter Financial News Sentiment
- **Nguồn**: [Hugging Face](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
- **Tổng số mẫu**: 2,388 (validation split)
- **Phân loại ban đầu**: 3 classes
  - Label 0 (Negative): 347 samples (14.53%)
  - Label 1 (Neutral): 475 samples (19.89%)
  - Label 2 (Positive): 1,566 samples (65.58%)
- **Sau khi chuyển đổi**: Binary classification
  - Negative/Neutral (0): 662 samples
  - Positive (1): 1,338 samples

## So sánh hiệu suất theo kích thước dataset

### Ex1: Small Dataset (100 samples)

| Metric | Training Set | Testing Set |
|--------|--------------|-------------|
| **Accuracy** | 96.00% | **84.00%** |
| **Precision** | 92.68% | 78.57% |
| **Recall** | 100.00% | 91.67% |
| **F1-Score** | 96.20% | 84.62% |

**Phân tích:**
- **Overfitting gap: 12.00%** (Model đang overfit)
- Vocabulary size: 614 từ
- Training time: 0.01s (10,617 samples/s)
- Misclassified: 4/25 samples (16%)

### Ex2: Medium Dataset (500 samples)

| Metric | Training Set | Testing Set |
|--------|--------------|-------------|
| **Accuracy** | 95.20% | **76.00%** |
| **Precision** | 92.08% | 70.00% |
| **Recall** | 98.94% | 90.32% |
| **F1-Score** | 95.38% | 78.87% |

**Phân tích:**
- **Overfitting gap: 19.20%** (Overfitting nghiêm trọng hơn!)
- Vocabulary size: 2,376 từ
- Training time: 0.07s (5,524 samples/s)
- Misclassified: 30/125 samples (24%)

**Lưu ý**: Với dataset trung bình, overfitting tăng cao do vocabulary lớn hơn nhưng số lượng mẫu chưa đủ để học tốt.

### Ex3: Large Dataset (2,000 samples)

| Metric | Training Set | Testing Set |
|--------|--------------|-------------|
| **Accuracy** | 87.60% | **80.00%** |
| **Precision** | 85.20% | 78.40% |
| **Recall** | 98.61% | 96.71% |
| **F1-Score** | 91.41% | 86.60% |

**Phân tích:**
- **Overfitting gap: 7.60%** (Cải thiện đáng kể!)
- Vocabulary size: 6,307 từ
- Training time: 0.62s (2,415 samples/s)
- Misclassified: 100/500 samples (20%)

**Kết luận**: Model generalize tốt hơn nhiều với dataset lớn.

## Biểu đồ so sánh

### Test Accuracy theo kích thước dataset

```
100% |                                              
 90% |                                              
 80% | ████████████   ███████████   ████████████████
 70% |    84.00%        76.00%         80.00%      
 60% |                                              
 50% |                                              
 40% |                                              
 30% |                                              
 20% |                                              
 10% |                                              
  0% +--------+------------+-----------+------------
       Small        Medium        Large             
     (100)         (500)        (2000)              
```

### Overfitting Gap theo kích thước dataset

```
25% |                                              
20% |              ████████████                     
15% |    ████████  19.20%                          
10% |    12.00%                ███████             
 5% |                          7.60%               
 0% +--------+------------+-----------+------------
       Small        Medium        Large             
```

## Kết luận chính

### 1. **Ảnh hưởng của kích thước dataset đến hiệu suất**

| Dataset Size | Test Accuracy | Overfitting Gap | Generalization |
|--------------|---------------|-----------------|----------------|
| Small (100)  | 84.00% | 12.00% | Moderate |
| Medium (500) | 76.00% | 19.20% | Poor |
| Large (2000) | **80.00%** | **7.60%** | Good |

**Nhận xets**: 
- Dataset nhỏ (100) có test accuracy cao nhất (84%) nhưng có overfitting
- Dataset trung bình (500) cho kết quả **tệ nhất** do overfitting nghiêm trọng
- Dataset lớn (2000) có test accuracy 80% và generalization **tốt nhất**

### 2. **Thông tin thêm**

Với Logistic Regression và TF-IDF:
- **< 200 samples**: Model đơn giản, có thể học tốt trên tập nhỏ nhưng không robust
- **200-1000 samples**: "Vùng nguy hiểm" - vocabulary lớn nhưng data chưa đủ → overfitting cao
- **> 1000 samples**: Model bắt đầu generlize tốt, overfitting giảm đáng kể

### 3. **Vocabulary Size Impact**

| Dataset | Vocabulary Size | Test Accuracy |
|---------|----------------|---------------|
| Small   | 614            | 84.00%        |
| Medium  | 2,376          | 76.00%        |
| Large   | 6,307          | 80.00%        |

**Nhận xét**: 
- Vocabulary tăng gấp 10 lần (614 → 6,307) với dataset lớn
- Nhiều features hơn giúp model học được patterns phức tạp hơn
- Cần đủ dữ liệu để "fill" vocabulary space

### 4. **Training Time vs Performance Trade-off**

| Dataset | Training Time | Samples/sec | Test Accuracy |
|---------|--------------|-------------|---------------|
| Small   | 0.01s        | 10,617      | 84.00%        |
| Medium  | 0.07s        | 5,524       | 76.00%        |
| Large   | 0.62s        | 2,415       | 80.00%        |

**Trade-off**: Thời gian training tăng ~60x nhưng model quality cải thiện đáng kể.

### Để cải thiện model performance:

1. **Thu thập thêm dữ liệu chất lượng cao**
   - Mục tiêu: > 2,000 samples với labels chính xác
   - Đảm bảo balance giữa các classes

2. **Cải thiện preprocessing**
   - Remove stop words
   - Stemming/Lemmatization
   - Handle domain-specific terms (financial terms)

3. **Feature Engineering**
   - N-grams (bigrams, trigrams)
   - Character-level features
   - Domain-specific features (e.g., stock symbols, financial keywords)

4. **Thử các models phức tạp hơn**
   - Random Forest
   - Gradient Boosting (XGBoost, LightGBM)
   - Neural Networks (LSTM, BERT)

5. **Hyperparameter Tuning**
   - TF-IDF parameters (max_df, min_df, ngram_range)
   - Logistic Regression parameters (C, penalty)
   - Cross-validation để tìm optimal parameters

6. **Regularization**
   - L1/L2 regularization để giảm overfitting
   - Feature selection để loại bỏ noise

## Ví dụ Misclassifications

### Các trường hợp model dự đoán sai:

1. **"$STML: Alliance Global Partners starts at Buy"**
   - True: Negative | Predicted: Positive
   - Lý do: Từ "Buy" có sentiment tích cực nhưng context là financial news

2. **"Diesel Demand Slump Signals Manufacturing Recession"**
   - True: Negative | Predicted: Positive
   - Lý do: Model chưa học được "Slump" và "Recession" là negative indicators

3. **"Central bank 'collateral damage' is skewing financial markets"**
   - True: Negative | Predicted: Positive
   - Lý do: "collateral damage" là cụm từ phức tạp cần context

### Patterns của misclassifications:

- **Financial jargon**: Model chưa hiểu rõ thuật ngữ tài chính
- **Sarcasm/Irony**: Khó phát hiện trong sentiment analysis
- **Mixed sentiment**: Câu chứa cả positive và negative terms
- **Context dependency**: Cần hiểu context chứ không chỉ keywords

## 4. Thách thức và Giải pháp

### Thách thức thứ nhất: Overfitting với Medium Dataset
**Vấn đề**: Dataset 500 samples có overfitting gap cao nhất (19.20%)
- Vocabulary tăng lên 2,376 words
- Model học quá tốt trên training set (95.2%) nhưng test set chỉ 76%

**giải pháp**:
- Tăng dataset size lên 2,000 samples → giảm overfitting xuống 7.6%
- Sử dụng regularization trong LogisticRegression
- Stratified train/test split để đảm bảo balance

### Thách thức thứ 2: Dataset không cân bằng
**Vấn đề**: Dataset gốc có 65.58% positive, chỉ 14.53% negative
- Model có thể bias về class positive

**Giải pháp**:
- Convert sang binary classification (positive vs non-positive)
- Sử dụng `stratify` parameter trong train_test_split
- Create balanced subsets với equal representation (50-50)

### Thách thức thứ 3: Financial Jargon và Context
**Vấn đè**: Model misclassify financial terms
- "$STML: Alliance Global Partners starts at Buy" → predicted Positive (should be Negative)
- "Buy", "bullish" có positive sentiment nhưng trong financial context khác

**Giải pháp**:
- Cần domain-specific preprocessing
- Có thể add financial keywords vào feature engineering
- Consider using embeddings trained on financial corpus

### Thách thức thứ 4: Small Vocabulary với Small Dataset
**Vấn đề**: 100 samples chỉ có 614 unique words
- Không đủ để capture diverse patterns
- Model có thể miss important terms

**Giải pháp**:
- Sử dụng larger dataset (2000 samples → 6,307 words)
- Vocabulary tăng 10x giúp model học được patterns phức tạp hơn

---

## 5. Kết luận cuối cùng

### Càng nhiều dữ liệu chất lượng, nhãn tốt thì model càng hiệu quả? Có, nhưng có điều kiện!

**Đúng khi:**
- Dữ liệu đủ lớn (> 1000-2000 samples)
- Labels chính xác và consistent
- Dataset balanced hoặc properly weighted
- Features relevant với task

**Cần lưu ý:**
- Dataset trung bình có thể tệ hơn dataset nhỏ do overfitting
- Quality > Quantity: 1000 samples tốt > 5000 samples noisy
- Cần balance giữa training time và performance
- Model architecture phải phù hợp với data size

### Metrics Comparison Summary

```
                Small    Medium    Large
                (100)    (500)    (2000)
Accuracy        84%      76%      80%     ← Large wins in generalization
Precision       79%      70%      78%
Recall          92%      90%      97%     ← Large highest recall
F1-Score        85%      79%      87%     ← Large best overall
Overfitting     12%      19%      8%      ← Large least overfitting
```

---
**Lưu ý:** Một số dữ liệu mãu được tạo bằng chat gpt để test code & fix bug - không được thống kê trong báo cáo này.
