import os
import json
import random
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from gensim.models import Word2Vec

# Paths
BASE = os.path.dirname(__file__)
ENC_DIR = os.path.join(BASE, 'hwu_encoded')
MODEL_DIR = os.path.join(BASE, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
W2V_PATH = os.path.join(MODEL_DIR, 'w2v.model')

# Hyperparams
EMB_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 6
MAX_LEN = 30
SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load data
train_path = os.path.join(ENC_DIR, 'train_encoded.csv')
val_path = os.path.join(ENC_DIR, 'val_encoded.csv')
test_path = os.path.join(ENC_DIR, 'test_encoded.csv')

print('Loading data...')
df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
df_test = pd.read_csv(test_path)

# Tokenizer
import re

def tokenize(text):
    if pd.isna(text):
        return []
    text = str(text).lower()
    # simple tokenization: split on whitespace and punctuation
    tokens = re.findall(r"\w+", text)
    return tokens

# Build vocab from train only
print('Building vocabulary...')
all_tokens = []
for t in df_train['text'].fillna(''):
    all_tokens.extend(tokenize(t))

counter = Counter(all_tokens)
# set vocab size to include all words with min freq 1 (or limit to top_k)
min_freq = 1
vocab_tokens = [w for w,c in counter.items() if c >= min_freq]
# special tokens
PAD = '<pad>'
UNK = '<unk>'
itos = [PAD, UNK] + vocab_tokens
stoi = {w:i for i,w in enumerate(itos)}
vocab_size = len(itos)
print('Vocab size:', vocab_size)

# Load pretrained Word2Vec (trained earlier) if exists
if os.path.exists(W2V_PATH):
    print('Loading Word2Vec as pretrained embeddings...')
    w2v = Word2Vec.load(W2V_PATH)
else:
    print('No pretrained Word2Vec found, training a fresh one on corpus...')
    corpus = [tokenize(t) for t in pd.concat([df_train['text'], df_val['text'], df_test['text']]).fillna('')]
    w2v = Word2Vec(sentences=corpus, vector_size=EMB_DIM, window=5, min_count=1, workers=4, epochs=10)
    w2v.save(W2V_PATH)

# Build embedding matrix from w2v for tokens in our vocab
def build_embedding_matrix(stoi, w2v_model, emb_dim):
    matrix = np.random.normal(scale=0.6, size=(len(stoi), emb_dim)).astype(np.float32)
    # set PAD vector to zeros
    matrix[stoi[PAD]] = np.zeros(emb_dim, dtype=np.float32)
    for token, idx in stoi.items():
        if token in w2v_model.wv:
            matrix[idx] = w2v_model.wv[token]
    return matrix

embedding_matrix = build_embedding_matrix(stoi, w2v, EMB_DIM)

# Dataset
class HWUDataset(Dataset):
    def __init__(self, texts, labels, stoi, max_len):
        self.texts = texts
        self.labels = labels
        self.stoi = stoi
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def text_to_indices(self, text):
        toks = tokenize(text)
        idxs = [self.stoi.get(t, self.stoi[UNK]) for t in toks][:self.max_len]
        # pad
        if len(idxs) < self.max_len:
            idxs = idxs + [self.stoi[PAD]] * (self.max_len - len(idxs))
        return np.array(idxs, dtype=np.int64)
    def __getitem__(self, idx):
        x = self.text_to_indices(self.texts.iloc[idx])
        y = int(self.labels.iloc[idx])
        return torch.tensor(x), torch.tensor(y)

train_ds = HWUDataset(df_train['text'].fillna(''), df_train['intent_id'], stoi, MAX_LEN)
val_ds = HWUDataset(df_val['text'].fillna(''), df_val['intent_id'], stoi, MAX_LEN)
test_ds = HWUDataset(df_test['text'].fillna(''), df_test['intent_id'], stoi, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

num_classes = len(pd.read_json(os.path.join(BASE, 'hwu_label_mapping.json'), typ='series'))
print('Num classes:', num_classes)

# Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, embedding_matrix=None, train_emb=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = train_emb
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, x):
        emb = self.embedding(x)
        out, (hn, cn) = self.lstm(emb)  # out: [B, L, H*2]
        # use mean pooling over time
        out = out.mean(dim=1)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


def train_model(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_f1 = 0.0
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        # validate
        val_metrics = eval_model(model, val_loader, device)
        print(f'Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - val_acc: {val_metrics["acc"]:.4f} - val_f1: {val_metrics["f1"]:.4f}')
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def eval_model(model, loader, device):
    model.eval()
    ys, ys_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            ys_pred.extend(preds.tolist())
            ys.extend(yb.numpy().tolist())
    acc = accuracy_score(ys, ys_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(ys, ys_pred, average='weighted', zero_division=0)
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}

# Train model 1: pretrained embeddings (from w2v) + LSTM
print('\n=== Model A: pre-trained embeddings (from Word2Vec) + LSTM ===')
model_a = LSTMClassifier(vocab_size=vocab_size, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, num_classes=num_classes, embedding_matrix=embedding_matrix, train_emb=True)
model_a = train_model(model_a, train_loader, val_loader, EPOCHS, LR, device)
val_metrics_a = eval_model(model_a, val_loader, device)
test_metrics_a = eval_model(model_a, test_loader, device)
print('Model A - Val:', val_metrics_a)
print('Model A - Test:', test_metrics_a)

torch.save(model_a.state_dict(), os.path.join(MODEL_DIR, 'lstm_pretrained.pt'))

# Train model 2: random embedding (train from scratch) + LSTM
print('\n=== Model B: random embeddings (train-from-scratch) + LSTM ===')
model_b = LSTMClassifier(vocab_size=vocab_size, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, num_classes=num_classes, embedding_matrix=None, train_emb=True)
model_b = train_model(model_b, train_loader, val_loader, EPOCHS, LR, device)
val_metrics_b = eval_model(model_b, val_loader, device)
test_metrics_b = eval_model(model_b, test_loader, device)
print('Model B - Val:', val_metrics_b)
print('Model B - Test:', test_metrics_b)

torch.save(model_b.state_dict(), os.path.join(MODEL_DIR, 'lstm_scratch.pt'))

# Save report
report = {
    'model_a': {'val': val_metrics_a, 'test': test_metrics_a},
    'model_b': {'val': val_metrics_b, 'test': test_metrics_b},
    'params': {'emb_dim': EMB_DIM, 'hidden_dim': HIDDEN_DIM, 'max_len': MAX_LEN}
}
with open(os.path.join(MODEL_DIR, 'lstm_report.json'), 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print('Saved LSTM models and report to', MODEL_DIR)
