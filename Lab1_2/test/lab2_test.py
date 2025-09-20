from Lab1_2.src.preprocessing.regex_tokenizer import RegexTokenizer
from Lab1_2.src.representations.count_vectorizer import CountVectorizer

corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]

tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer)

vectors = vectorizer.fit_transform(corpus)

print("Vocabulary:")
print(vectorizer.vocabulary_)
print("\nDocument-term matrix:")
for vec in vectors:
    print(vec)
