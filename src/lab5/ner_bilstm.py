import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

SEED = 42
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
EPOCHS = 5
LEARNING_RATE = 1e-3
MAX_EXAMPLES = 5

OUTPUT_DIR = Path('Lab5_part4/results')
MODEL_DIR = Path('Lab5_part4/models')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_conll2003():
    dataset = load_dataset('conll2003', trust_remote_code=True)
    tag_names = dataset['train'].features['ner_tags'].feature.names

    def convert_split(split: str) -> Tuple[List[List[str]], List[List[str]]]:
        tokens = dataset[split]['tokens']
        tag_ids = dataset[split]['ner_tags']
        tag_strings = [[tag_names[idx] for idx in seq] for seq in tag_ids]
        return tokens, tag_strings

    splits = {
        'train': convert_split('train'),
        'validation': convert_split('validation'),
        'test': convert_split('test'),
    }
    return splits, tag_names


def build_word_vocab(sentences: Sequence[Sequence[str]]) -> Dict[str, int]:
    word_to_ix = {'<PAD>': 0, '<UNK>': 1}
    for sent in sentences:
        for token in sent:
            if token not in word_to_ix:
                word_to_ix[token] = len(word_to_ix)
    return word_to_ix


def build_tag_vocab(tag_names: Sequence[str]) -> Dict[str, int]:
    tag_to_ix = {tag: idx for idx, tag in enumerate(tag_names)}
    if '<PAD>' not in tag_to_ix:
        tag_to_ix['<PAD>'] = len(tag_to_ix)
    return tag_to_ix


class NERDataset(Dataset):
    def __init__(self, sentences: Sequence[Sequence[str]], tags: Sequence[Sequence[str]], word_to_ix: Dict[str, int], tag_to_ix: Dict[str, int]):
        self.sentences = sentences
        self.tags = tags
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.unk_idx = word_to_ix['<UNK>']

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        tokens = self.sentences[idx]
        ner_tags = self.tags[idx]
        word_indices = [self.word_to_ix.get(tok, self.unk_idx) for tok in tokens]
        tag_indices = [self.tag_to_ix[tag] for tag in ner_tags]
        return torch.tensor(word_indices, dtype=torch.long), torch.tensor(tag_indices, dtype=torch.long)


def make_collate_fn(pad_word_idx: int, pad_tag_idx: int):
    def collate(batch):
        word_seqs, tag_seqs = zip(*batch)
        padded_words = pad_sequence(word_seqs, batch_first=True, padding_value=pad_word_idx)
        padded_tags = pad_sequence(tag_seqs, batch_first=True, padding_value=pad_tag_idx)
        lengths = torch.tensor([len(seq) for seq in word_seqs], dtype=torch.long)
        return padded_words, padded_tags, lengths

    return collate


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size: int, tagset_size: int, embedding_dim: int, hidden_dim: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        outputs, _ = self.lstm(emb)
        logits = self.fc(outputs)
        return logits


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, pad_tag_idx: int) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for words, tags, _ in dataloader:
        words = words.to(device)
        tags = tags.to(device)
        optimizer.zero_grad()
        logits = model(words)
        loss = criterion(logits.view(-1, logits.size(-1)), tags.view(-1))
        loss.backward()
        optimizer.step()
        valid_tokens = (tags != pad_tag_idx).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens
    return total_loss / max(total_tokens, 1)


def evaluate(model: nn.Module, dataloader: DataLoader, pad_tag_idx: int, ix_to_word: Dict[int, str] | None = None, ix_to_tag: Dict[int, str] | None = None, collect_examples: bool = False):
    model.eval()
    total_correct = 0
    total_tokens = 0
    collected = []
    need_examples = collect_examples
    with torch.no_grad():
        for words, tags, lengths in dataloader:
            words = words.to(device)
            tags = tags.to(device)
            logits = model(words)
            preds = logits.argmax(dim=-1)
            mask = tags != pad_tag_idx
            total_correct += (preds[mask] == tags[mask]).sum().item()
            total_tokens += mask.sum().item()
            if need_examples and ix_to_word and ix_to_tag:
                batch_size = words.size(0)
                for i in range(batch_size):
                    sent_len = lengths[i].item()
                    token_ids = words[i, :sent_len].cpu().tolist()
                    pred_ids = preds[i, :sent_len].cpu().tolist()
                    gold_ids = tags[i, :sent_len].cpu().tolist()
                    tokens = [ix_to_word[idx] for idx in token_ids]
                    pred_tags = [ix_to_tag[idx] for idx in pred_ids]
                    gold_tags = [ix_to_tag[idx] for idx in gold_ids]
                    collected.append({'tokens': tokens, 'pred_tags': pred_tags, 'gold_tags': gold_tags})
                    if len(collected) >= MAX_EXAMPLES:
                        need_examples = False
                        break
    accuracy = total_correct / max(total_tokens, 1)
    return accuracy, collected


def predict_sentence(model: nn.Module, sentence: str, word_to_ix: Dict[str, int], ix_to_tag: Dict[int, str]):
    tokens = sentence.strip().split()
    indices = [word_to_ix.get(tok, word_to_ix['<UNK>']) for tok in tokens]
    tensor = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        preds = logits.argmax(dim=-1).squeeze(0).tolist()
    return list(zip(tokens, [ix_to_tag[idx] for idx in preds]))


def main():
    splits, tag_names = load_conll2003()
    train_sentences, train_tags = splits['train']
    val_sentences, val_tags = splits['validation']
    test_sentences, test_tags = splits['test']

    word_to_ix = build_word_vocab(train_sentences)
    tag_to_ix = build_tag_vocab(tag_names)

    ix_to_word = {idx: word for word, idx in word_to_ix.items()}
    ix_to_tag = {idx: tag for tag, idx in tag_to_ix.items()}

    train_dataset = NERDataset(train_sentences, train_tags, word_to_ix, tag_to_ix)
    val_dataset = NERDataset(val_sentences, val_tags, word_to_ix, tag_to_ix)
    test_dataset = NERDataset(test_sentences, test_tags, word_to_ix, tag_to_ix)

    pad_word_idx = word_to_ix['<PAD>']
    pad_tag_idx = tag_to_ix['<PAD>']
    collate_fn = make_collate_fn(pad_word_idx, pad_tag_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = BiLSTMTagger(
        vocab_size=len(word_to_ix),
        tagset_size=len(tag_to_ix),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        pad_idx=pad_word_idx,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_tag_idx)

    best_dev = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, pad_tag_idx)
        train_acc, _ = evaluate(model, train_loader, pad_tag_idx)
        val_acc, _ = evaluate(model, val_loader, pad_tag_idx)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_acc})
        print(f'Epoch {epoch}/{EPOCHS} | loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f}')
        if val_acc > best_dev:
            best_dev = val_acc
            torch.save(model.state_dict(), MODEL_DIR / 'ner_bilstm.pt')
            print(f'New best validation accuracy: {val_acc:.4f} (model saved)')

    best_model = BiLSTMTagger(
        vocab_size=len(word_to_ix),
        tagset_size=len(tag_to_ix),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        pad_idx=pad_word_idx,
    ).to(device)
    best_model.load_state_dict(torch.load(MODEL_DIR / 'ner_bilstm.pt', map_location=device))

    train_acc, _ = evaluate(best_model, train_loader, pad_tag_idx)
    val_acc, examples = evaluate(best_model, val_loader, pad_tag_idx, ix_to_word=ix_to_word, ix_to_tag=ix_to_tag, collect_examples=True)
    test_acc, _ = evaluate(best_model, test_loader, pad_tag_idx)

    print(f'Final accuracy -> Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}')

    report = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'history': history,
        'hyperparams': {
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
        }
    }
    with open(OUTPUT_DIR / 'ner_bilstm_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / 'ner_bilstm_examples.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    sample_sentence = 'VNU University is located in Hanoi'
    sample_prediction = predict_sentence(best_model, sample_sentence, word_to_ix, ix_to_tag)
    with open(OUTPUT_DIR / 'ner_sample_prediction.json', 'w', encoding='utf-8') as f:
        json.dump({'sentence': sample_sentence, 'prediction': sample_prediction}, f, ensure_ascii=False, indent=2)
    print(f"Sample prediction for '{sample_sentence}': {sample_prediction}")


if __name__ == '__main__':
    main()
