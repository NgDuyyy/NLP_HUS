from Lab1_2.src.core.interfaces import Tokenizer
import re

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        # Convert to lowercase
        text = text.lower()
        # Split punctuation from words
        text = re.sub(r'([.,?!])', r' \1 ', text)
        # Split by whitespace
        tokens = text.split()
        return tokens
