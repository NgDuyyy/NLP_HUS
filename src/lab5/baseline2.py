import os
import json
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

BASE = os.path.dirname(__file__)
ENC_DIR = os.path.join(BASE, 'hwu_encoded')
MODEL_DIR = os.path.join(BASE, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

train_path = os.path.join(ENC_DIR, 'train_encoded.csv')
val_path = os.path.join(ENC_DIR, 'val_encoded.csv')
test_path = os.path.join(ENC_DIR, 'test_encoded.csv')

print('Loading encoded CSVs...')
df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
df_test = pd.read_csv(test_path)

# Basic tokenization: simple whitespace + lowercase
def tokenize(text):
    if pd.isna(text):
        return []
    return [t for t in str(text).lower().split() if t]

# Prepare corpus for Word2Vec: train+val+test (to maximize coverage)
print('Tokenizing corpus for Word2Vec training...')
corpus = []
for df in [df_train, df_val, df_test]:
    corpus.extend([tokenize(t) for t in df['text'].fillna('')])

# Train Word2Vec
w2v_size = 100
print('Training Word2Vec model (this may take a moment)...')
w2v = Word2Vec(sentences=corpus, vector_size=w2v_size, window=5, min_count=1, workers=4, epochs=10)

# Function to get average vector for a text
def avg_vector(text_tokens, model):
    vecs = []
    for tok in text_tokens:
        if tok in model.wv:
            vecs.append(model.wv[tok])
    if len(vecs) == 0:
        return np.zeros(model.vector_size, dtype=float)
    return np.mean(vecs, axis=0)

# Build feature matrices
print('Building averaged vectors for train/val/test...')
X_train = np.vstack([avg_vector(tokenize(t), w2v) for t in df_train['text'].fillna('')])
y_train = df_train['intent_id'].values

X_val = np.vstack([avg_vector(tokenize(t), w2v) for t in df_val['text'].fillna('')])
y_val = df_val['intent_id'].values

X_test = np.vstack([avg_vector(tokenize(t), w2v) for t in df_test['text'].fillna('')])
y_test = df_test['intent_id'].values

print('Shapes:', X_train.shape, X_val.shape, X_test.shape)

# Train Logistic Regression
print('Training Logistic Regression on averaged Word2Vec features...')
clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga', multi_class='multinomial')
clf.fit(X_train, y_train)

# Evaluate
print('Evaluating on validation set...')
val_pred = clf.predict(X_val)
acc = accuracy_score(y_val, val_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_val, val_pred, average='weighted', zero_division=0)
print(f'Validation - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}')

print('Evaluating on test set...')
test_pred = clf.predict(X_test)
acc_t = accuracy_score(y_test, test_pred)
prec_t, rec_t, f1_t, _ = precision_recall_fscore_support(y_test, test_pred, average='weighted', zero_division=0)
print(f'Test - Acc: {acc_t:.4f}, Prec: {prec_t:.4f}, Rec: {rec_t:.4f}, F1: {f1_t:.4f}')

# Save models and report
w2v.save(os.path.join(MODEL_DIR, 'w2v.model'))
joblib.dump(clf, os.path.join(MODEL_DIR, 'logreg_w2v_mean.joblib'))
print('Saved Word2Vec model and classifier to', MODEL_DIR)

report = {
    'val': {'acc': float(acc), 'prec': float(prec), 'rec': float(rec), 'f1': float(f1)},
    'test': {'acc': float(acc_t), 'prec': float(prec_t), 'rec': float(rec_t), 'f1': float(f1_t)},
    'w2v_size': w2v_size,
}
with open(os.path.join(MODEL_DIR, 'baseline2_report.json'), 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print('Saved report to baseline2_report.json')
