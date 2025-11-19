import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np

DATA_DIR = Path('data/UD_English-EWT-r1.0')
TRAIN_FILE = DATA_DIR / 'en-ud-train.conllu'
DEV_FILE = DATA_DIR / 'en-ud-dev.conllu'
TEST_FILE = DATA_DIR / 'en-ud-test.conllu'

OUTPUT_DIR = Path('Lab5_part3/results')
MODEL_DIR = Path('Lab5_part3/models')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EMB_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-3
LOWERCASE = True
MAX_PRED_SENTENCES = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_conllu(file_path: Path) -> List[List[Tuple[str, str]]]:
    sentences = []
    current = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            token_id = parts[0]
            # Skip multi-word tokens like 1-2, etc.
            if '-' in token_id or '.' in token_id:
                continue
            word = parts[1]
            upos = parts[3]
            current.append((word, upos))
    if current:
        sentences.append(current)
    return sentences


def build_vocab(sentences: List[List[Tuple[str, str]]], lowercase: bool = True) -> Tuple[Dict[str, int], Dict[str, int]]:
    word_to_ix = {'<PAD>': 0, '<UNK>': 1}
    tag_to_ix = {'<PAD>': 0}
    for sent in sentences:
        for word, tag in sent:
            key = word.lower() if lowercase else word
            if key not in word_to_ix:
                word_to_ix[key] = len(word_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    return word_to_ix, tag_to_ix


class POSDataset(Dataset):
    def __init__(self, sentences: List[List[Tuple[str, str]]], word_to_ix: Dict[str, int], tag_to_ix: Dict[str, int], lowercase: bool = True):
        self.sentences = sentences
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.lowercase = lowercase
        self.unk_idx = word_to_ix['<UNK>']

    def __len__(self):
        return len(self.sentences)

    def encode_sentence(self, sent: List[Tuple[str, str]]):
        words, tags = zip(*sent)
        if self.lowercase:
            word_indices = [self.word_to_ix.get(w.lower(), self.unk_idx) for w in words]
        else:
            word_indices = [self.word_to_ix.get(w, self.unk_idx) for w in words]
        tag_indices = [self.tag_to_ix[t] for t in tags]
        return torch.tensor(word_indices, dtype=torch.long), torch.tensor(tag_indices, dtype=torch.long)

    def __getitem__(self, idx):
        return self.encode_sentence(self.sentences[idx])


def make_collate_fn(pad_word_idx: int, pad_tag_idx: int):
    def collate(batch):
        word_seqs, tag_seqs = zip(*batch)
        word_padded = pad_sequence(word_seqs, batch_first=True, padding_value=pad_word_idx)
        tag_padded = pad_sequence(tag_seqs, batch_first=True, padding_value=pad_tag_idx)
        lengths = torch.tensor([len(seq) for seq in word_seqs], dtype=torch.long)
        return word_padded, tag_padded, lengths
    return collate


class SimpleRNNForTokenClassification(nn.Module):
    def __init__(self, vocab_size: int, tagset_size: int, embedding_dim: int, hidden_dim: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        emb = self.embedding(x)
        outputs, _ = self.rnn(emb)
        logits = self.fc(outputs)
        return logits


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    total_tokens = 0
    for words, tags, _ in dataloader:
        words = words.to(device)
        tags = tags.to(device)
        optimizer.zero_grad()
        logits = model(words)
        loss = criterion(logits.view(-1, logits.size(-1)), tags.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * tags.numel()
        total_tokens += tags.numel()
    return running_loss / total_tokens


def evaluate(model, dataloader, pad_tag_idx: int, collect_examples: bool = False):
    model.eval()
    total_correct = 0
    total_tokens = 0
    collected = []
    with torch.no_grad():
        for words, tags, lengths in dataloader:
            words = words.to(device)
            tags = tags.to(device)
            logits = model(words)
            preds = logits.argmax(dim=-1)
            mask = tags != pad_tag_idx
            total_correct += (preds[mask] == tags[mask]).sum().item()
            total_tokens += mask.sum().item()
            if collect_examples:
                for i in range(words.size(0)):
                    sent_len = lengths[i].item()
                    tok_preds = preds[i, :sent_len].cpu().tolist()
                    tok_gold = tags[i, :sent_len].cpu().tolist()
                    collected.append((words[i, :sent_len].cpu().tolist(), tok_preds, tok_gold))
                    if len(collected) >= MAX_PRED_SENTENCES:
                        collect_examples = False
                        break
            if not collect_examples and len(collected) >= MAX_PRED_SENTENCES:
                break
    accuracy = total_correct / max(total_tokens, 1)
    return accuracy, collected


def indices_to_words(indices: List[int], ix_to_word: Dict[int, str]):
    return [ix_to_word.get(idx, '<UNK>') for idx in indices]


def predict_sentence(model, sentence: str, word_to_ix: Dict[str, int], ix_to_tag: Dict[int, str]):
    tokens = sentence.strip().split()
    indices = [word_to_ix.get(tok.lower() if LOWERCASE else tok, word_to_ix['<UNK>']) for tok in tokens]
    tensor = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        preds = logits.argmax(dim=-1).squeeze(0).tolist()
    return list(zip(tokens, [ix_to_tag[idx] for idx in preds]))


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Cannot find UD dataset at {TRAIN_FILE}")

    print('Loading UD English-EWT dataset...')
    train_sentences = load_conllu(TRAIN_FILE)
    dev_sentences = load_conllu(DEV_FILE)
    test_sentences = load_conllu(TEST_FILE)
    print(f'Train sentences: {len(train_sentences)}')
    print(f'Dev sentences: {len(dev_sentences)}')
    print(f'Test sentences: {len(test_sentences)}')

    word_to_ix, tag_to_ix = build_vocab(train_sentences, lowercase=LOWERCASE)
    print(f'Vocabulary size: {len(word_to_ix)}')
    print(f'Number of POS tags: {len(tag_to_ix)}')

    ix_to_word = {idx: word for word, idx in word_to_ix.items()}
    ix_to_tag = {idx: tag for tag, idx in tag_to_ix.items()}

    train_ds = POSDataset(train_sentences, word_to_ix, tag_to_ix, lowercase=LOWERCASE)
    dev_ds = POSDataset(dev_sentences, word_to_ix, tag_to_ix, lowercase=LOWERCASE)
    test_ds = POSDataset(test_sentences, word_to_ix, tag_to_ix, lowercase=LOWERCASE)

    pad_word_idx = word_to_ix['<PAD>']
    pad_tag_idx = tag_to_ix['<PAD>']
    collate_fn = make_collate_fn(pad_word_idx, pad_tag_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = SimpleRNNForTokenClassification(
        vocab_size=len(word_to_ix),
        tagset_size=len(tag_to_ix),
        embedding_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        pad_idx=pad_word_idx,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_tag_idx)

    best_dev = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        train_acc, _ = evaluate(model, train_loader, pad_tag_idx)
        dev_acc, _ = evaluate(model, dev_loader, pad_tag_idx)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'dev_acc': dev_acc})
        print(f'Epoch {epoch}/{EPOCHS} | loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | dev_acc: {dev_acc:.4f}')
        if dev_acc > best_dev:
            best_dev = dev_acc
            torch.save(model.state_dict(), MODEL_DIR / 'pos_rnn.pt')
            print(f'New best dev accuracy: {dev_acc:.4f} (model saved)')

    # Load best model for evaluation
    best_model = SimpleRNNForTokenClassification(
        vocab_size=len(word_to_ix),
        tagset_size=len(tag_to_ix),
        embedding_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        pad_idx=pad_word_idx,
    ).to(device)
    best_model.load_state_dict(torch.load(MODEL_DIR / 'pos_rnn.pt', map_location=device))

    train_acc, _ = evaluate(best_model, train_loader, pad_tag_idx)
    dev_acc, examples = evaluate(best_model, dev_loader, pad_tag_idx, collect_examples=True)
    test_acc, _ = evaluate(best_model, test_loader, pad_tag_idx)

    print(f'Final accuracy -> Train: {train_acc:.4f} | Dev: {dev_acc:.4f} | Test: {test_acc:.4f}')

    report = {
        'train_acc': train_acc,
        'dev_acc': dev_acc,
        'test_acc': test_acc,
        'history': history,
        'hyperparams': {
            'embedding_dim': EMB_DIM,
            'hidden_dim': HIDDEN_DIM,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LR,
        }
    }

    with open(OUTPUT_DIR / 'pos_rnn_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Store qualitative examples
    qualitative = []
    for idx, (word_ids, pred_ids, gold_ids) in enumerate(examples):
        tokens = indices_to_words(word_ids, ix_to_word)
        pred_tags = [ix_to_tag[i] for i in pred_ids]
        gold_tags = [ix_to_tag[i] for i in gold_ids]
        qualitative.append({
            'sentence': ' '.join(tokens),
            'pred_tags': pred_tags,
            'gold_tags': gold_tags,
        })
    with open(OUTPUT_DIR / 'pos_rnn_examples.json', 'w', encoding='utf-8') as f:
        json.dump(qualitative, f, ensure_ascii=False, indent=2)

    # Predict sample sentence
    sample = 'I love NLP'
    prediction = predict_sentence(best_model, sample, word_to_ix, ix_to_tag)
    print(f"Sample prediction for '{sample}': {prediction}")
    with open(OUTPUT_DIR / 'pos_rnn_sample_prediction.json', 'w', encoding='utf-8') as f:
        json.dump({'sentence': sample, 'prediction': prediction}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
