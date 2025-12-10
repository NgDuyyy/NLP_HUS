import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, pipeline

OUTPUT_DIR = Path('result/lab6')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_fill_mask():
    mask_filler = pipeline('fill-mask', model='bert-base-uncased')
    sentence = 'Hanoi is the [MASK] of Vietnam.'
    predictions = mask_filler(sentence, top_k=5)
    has_capital = any(pred['token_str'].strip().lower() == 'capital' for pred in predictions)
    return {
        'task': 'fill-mask',
        'input': sentence,
        'predictions': predictions,
        'contains_capital': has_capital,
    }


def run_text_generation():
    generator = pipeline('text-generation', model='gpt2')
    prompt = 'The best thing about learning NLP is'
    outputs = generator(prompt, max_length=50, num_return_sequences=1, do_sample=False)
    return {
        'task': 'text-generation',
        'prompt': prompt,
        'outputs': outputs,
    }


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def run_sentence_embedding():
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    sentence = 'This is a sample sentence.'
    inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])[0]
    return {
        'task': 'sentence-embedding',
        'model_name': model_name,
        'sentence': sentence,
        'embedding_dim': embedding.shape[-1],
        'embedding': embedding.tolist(),
    }


def main():
    results = {
        'fill_mask': run_fill_mask(),
        'text_generation': run_text_generation(),
        'sentence_embedding': run_sentence_embedding(),
    }
    output_file = OUTPUT_DIR / 'lab6_part1_outputs.json'
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f'Results written to {output_file}')


if __name__ == '__main__':
    main()
