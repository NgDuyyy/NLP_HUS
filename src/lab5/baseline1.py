import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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

X_train, y_train = df_train['text'].fillna(''), df_train['intent_id']
X_val, y_val = df_val['text'].fillna(''), df_val['intent_id']
X_test, y_test = df_test['text'].fillna(''), df_test['intent_id']

print('Training TF-IDF vectorizer...')
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

print('Training Logistic Regression...')
clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga', multi_class='multinomial')
clf.fit(X_train_vec, y_train)

# Evaluate on validation
print('Evaluating on validation set...')
val_pred = clf.predict(X_val_vec)
acc = accuracy_score(y_val, val_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_val, val_pred, average='weighted', zero_division=0)
print(f'Validation - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}')

# Evaluate on test
print('Evaluating on test set...')
test_pred = clf.predict(X_test_vec)
acc_t = accuracy_score(y_test, test_pred)
prec_t, rec_t, f1_t, _ = precision_recall_fscore_support(y_test, test_pred, average='weighted', zero_division=0)
print(f'Test - Acc: {acc_t:.4f}, Prec: {prec_t:.4f}, Rec: {rec_t:.4f}, F1: {f1_t:.4f}')

# Save model and vectorizer
joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
joblib.dump(clf, os.path.join(MODEL_DIR, 'logreg_tfidf.joblib'))
print('Saved vectorizer and model to', MODEL_DIR)

# Save a short report
report = {
    'val': {'acc': float(acc), 'prec': float(prec), 'rec': float(rec), 'f1': float(f1)},
    'test': {'acc': float(acc_t), 'prec': float(prec_t), 'rec': float(rec_t), 'f1': float(f1_t)}
}
with open(os.path.join(MODEL_DIR, 'baseline1_report.json'), 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print('Saved report to baseline1_report.json')
