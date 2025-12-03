from abc import ABC, abstractmethod
from typing import List
class Vectorizer(ABC):
	@abstractmethod
	def fit(self, corpus: List[str]):
		pass

	@abstractmethod
	def transform(self, documents: List[str]) -> List[List[int]]:
		pass

	@abstractmethod
	def fit_transform(self, corpus: List[str]) -> List[List[int]]:
		pass

class Tokenizer(ABC):
	@abstractmethod
	def tokenize(self, text: str) -> List[str]:
		pass
