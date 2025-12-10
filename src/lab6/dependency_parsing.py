from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token

RESULTS_DIR = Path("result/lab6")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "lab6_dependency_parsing_outputs.json"


@lru_cache(maxsize=1)
def load_pipeline(model_name: str = "en_core_web_md") -> Language:
    """Load a spaCy pipeline with a graceful fallback when the default model is missing."""

    def _try_load(name: str) -> Language | None:
        try:
            return spacy.load(name)
        except OSError:
            try:
                from spacy.cli import download

                download(name)
                return spacy.load(name)
            except Exception:
                return None

    pipeline = _try_load(model_name)
    if pipeline is None and model_name != "en_core_web_sm":
        pipeline = _try_load("en_core_web_sm")

    if pipeline is None:
        raise RuntimeError(
            "No compatible spaCy model is installed. Run 'python -m spacy download en_core_web_md'."
        )
    return pipeline


def find_main_verb(doc: Doc) -> Token | None:
    """Return the main verb (ROOT) token of the sentence if it exists."""
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}:
            return token
    for token in doc:
        if token.dep_ == "ROOT":
            return token
    return None


def extract_noun_chunks(doc: Doc) -> List[Span]:
    """Custom noun chunk extraction mirroring spaCy's noun_chunks utility."""
    modifier_deps = {"amod", "compound", "det", "poss", "nummod", "quantmod"}
    valid_pos = {"NOUN", "PROPN"}
    spans: List[Span] = []
    seen: set[tuple[int, int]] = set()

    for token in doc:
        if token.pos_ not in valid_pos:
            continue
        modifiers = [child for child in token.children if child.dep_ in modifier_deps and child.i < token.i]
        span_tokens = sorted(modifiers + [token], key=lambda t: t.i)
        start, end = span_tokens[0].i, span_tokens[-1].i + 1
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        spans.append(doc[start:end])
    return spans


def get_path_to_root(token: Token) -> List[Token]:
    """Return the path from the provided token up to the ROOT token (inclusive)."""
    path: List[Token] = []
    current = token
    while True:
        path.append(current)
        if current.dep_ == "ROOT" or current.head == current:
            break
        current = current.head
    return path


def _serialize_span(span: Span) -> dict:
    return {"text": span.text, "start": span.start, "end": span.end}


def _serialize_path(path: Sequence[Token]) -> List[dict]:
    return [
        {
            "text": tok.text,
            "dep": tok.dep_,
            "pos": tok.pos_,
        }
        for tok in path
    ]


def run_examples() -> dict:
    nlp = load_pipeline()

    main_verb_text = "The cat chased the mouse and the dog watched them."
    main_verb_doc = nlp(main_verb_text)
    main_verb_token = find_main_verb(main_verb_doc)

    noun_chunk_text = "The big, fluffy white cat is sleeping on the warm mat."
    noun_chunk_doc = nlp(noun_chunk_text)
    noun_chunks = extract_noun_chunks(noun_chunk_doc)

    path_text = "Apple is looking at buying U.K. startup for $1 billion."
    path_doc = nlp(path_text)
    startup_token = next((tok for tok in path_doc if tok.text.lower() == "startup"), path_doc[0])
    path_tokens = get_path_to_root(startup_token)

    return {
        "main_verb_example": {
            "text": main_verb_text,
            "main_verb": main_verb_token.text if main_verb_token else None,
            "dependency": main_verb_token.dep_ if main_verb_token else None,
        },
        "noun_chunk_example": {
            "text": noun_chunk_text,
            "chunks": [_serialize_span(span) for span in noun_chunks],
        },
        "path_example": {
            "text": path_text,
            "token": startup_token.text,
            "path": _serialize_path(path_tokens),
        },
    }


def main() -> None:
    results = run_examples()
    RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dependency parsing exercises exported to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
