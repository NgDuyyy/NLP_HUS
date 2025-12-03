from Lab1_2.src.core.interfaces import Tokenizer
import re

class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        # Sử dụng regex để tách từ và dấu câu
        return re.findall(r"\w+|[^\w\s]", text)
