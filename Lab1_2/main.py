from Lab1_2.src.preprocessing.simple_tokenizer import SimpleTokenizer
from Lab1_2.src.preprocessing.regex_tokenizer import RegexTokenizer
from Lab1_2.src.core.dataset_loaders import load_raw_text_data

sentences = [
    "Hello, world! This is a test.",
    "NLP is fascinating... isn't it?",
    "Let's see how it handles 123 numbers and punctuation!"
]

def test_tokenizer(tokenizer, name):
    print(f"\n{name}:")
    for s in sentences:
        print(f"Input: {s}")
        print(f"Tokens: {tokenizer.tokenize(s)}")


if __name__ == "__main__":
    test_tokenizer(SimpleTokenizer(), "SimpleTokenizer")
    test_tokenizer(RegexTokenizer(), "RegexTokenizer")

    # Task 3: Tokenization with UD_English-EWT Dataset
    dataset_path = "E:\\HUS_Y3\\NLP\\Lab1_2\\src\\core\\UD_English-EWT\\en-ud-train.conllu"
    try:
        raw_text = load_raw_text_data(dataset_path)
        sample_text = raw_text[:500]
        print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
        print(f"Original Sample: {sample_text[:100]}...")

        simple_tokenizer = SimpleTokenizer()
        regex_tokenizer = RegexTokenizer()

        simple_tokens = simple_tokenizer.tokenize(sample_text)
        print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")

        regex_tokens = regex_tokenizer.tokenize(sample_text)
        print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")
    except Exception as e:
        print(f"Không thể đọc dữ liệu: {e}")
