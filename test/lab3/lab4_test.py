import os
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Lab3_part2.src.representations.word_embedder import WordEmbedder


def print_vector(name: str, vector, decimals: int = 4, limit: int = 10) -> None:
    """Utility to pretty-print vectors."""

    if vector is None:
        print(f"{name}: <OOV>")
        return
    with np.printoptions(precision=decimals, suppress=True, linewidth=120):
        preview = vector[:limit]
        print(f"{name} (first {len(preview)} dims): {preview}")


def main() -> None:
    embedder = WordEmbedder()

    # Vector for 'king'
    king_vector = embedder.get_vector("king")
    print_vector("Vector for 'king'", king_vector)

    # Similarities
    sim_king_queen = embedder.get_similarity("king", "queen")
    sim_king_man = embedder.get_similarity("king", "man")
    print(f"Similarity(king, queen): {sim_king_queen:.4f}")
    print(f"Similarity(king, man):   {sim_king_man:.4f}")

    # Most similar to 'computer'
    similar_words = embedder.get_most_similar("computer", top_n=10)
    print("\nTop 10 words similar to 'computer':")
    for idx, sw in enumerate(similar_words, 1):
        print(f"  {idx:2d}. {sw.word:15s} -> {sw.score:.4f}")

    # Document embedding
    document = "The queen rules the country."
    doc_vector = embedder.embed_document(document)
    print_vector("Document embedding", doc_vector)


if __name__ == "__main__":
    main()
