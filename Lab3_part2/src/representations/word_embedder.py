from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import gensim.downloader as api
import numpy as np
from gensim.models import KeyedVectors

from Lab1_2.src.preprocessing.simple_tokenizer import SimpleTokenizer


@dataclass
class SimilarWord:
    """Structure to hold similar word and its score."""

    word: str
    score: float


class WordEmbedder:
    """Utility class for working with pre-trained word embeddings."""

    def __init__(self, model_name: str = "glove-wiki-gigaword-50", tokenizer: Optional[SimpleTokenizer] = None) -> None:
        self.model_name = model_name
        self.model: KeyedVectors = api.load(model_name)
        self.vector_size = self.model.vector_size
        self.tokenizer = tokenizer or SimpleTokenizer()

    def _is_in_vocab(self, word: str) -> bool:
        return word in self.model

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Return the embedding vector for a word, handling OOV cases."""

        word = word.lower()
        if not self._is_in_vocab(word):
            return None
        return self.model[word]

    def get_similarity(self, word1: str, word2: str) -> Optional[float]:
        """Return cosine similarity between two words if both exist."""

        word1 = word1.lower()
        word2 = word2.lower()
        if not (self._is_in_vocab(word1) and self._is_in_vocab(word2)):
            return None
        return float(self.model.similarity(word1, word2))

    def get_most_similar(self, word: str, top_n: int = 10) -> List[SimilarWord]:
        """Return top-N most similar words for the given input word."""

        word = word.lower()
        if not self._is_in_vocab(word):
            return []
        results = self.model.most_similar(word, topn=top_n)
        return [SimilarWord(w, float(score)) for w, score in results]

    def embed_document(self, document: str) -> np.ndarray:
        """Embed a whole document by averaging known word vectors."""

        tokens = self.tokenizer.tokenize(document)
        vectors: List[np.ndarray] = []

        for token in tokens:
            vec = self.get_vector(token)
            if vec is not None:
                vectors.append(vec)

        if not vectors:
            return np.zeros(self.vector_size, dtype=float)
        return np.mean(np.stack(vectors), axis=0)

    def embed_tokens(self, tokens: Iterable[str]) -> np.ndarray:
        """Embed a pre-tokenised iterable of words."""

        vectors: List[np.ndarray] = []
        for token in tokens:
            vec = self.get_vector(token)
            if vec is not None:
                vectors.append(vec)
        if not vectors:
            return np.zeros(self.vector_size, dtype=float)
        return np.mean(np.stack(vectors), axis=0)
