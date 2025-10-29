import math
from typing import List, Dict
from abc import ABC, abstractmethod


class Vectorizer(ABC):
    """Abstract base class for vectorizers"""
    @abstractmethod
    def fit(self, corpus: List[str]):
        pass

    @abstractmethod
    def transform(self, documents: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def fit_transform(self, corpus: List[str]) -> List[List[float]]:
        pass


class TfidfVectorizer(Vectorizer):
    """
    TF-IDF Vectorizer that converts a collection of text documents 
    to a matrix of TF-IDF features.
    """
    
    def __init__(self, tokenizer):
        """
        Initialize the TfidfVectorizer.
        
        Args:
            tokenizer: A tokenizer object with a tokenize() method
        """
        self.tokenizer = tokenizer
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Dict[str, float] = {}
        self.n_documents = 0
    
    def fit(self, corpus: List[str]):
        """
        Learn vocabulary and IDF from training corpus.
        
        Args:
            corpus: List of text documents
        """
        # Build vocabulary
        unique_tokens = set()
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            unique_tokens.update(tokens)
        
        # Assign index to each token
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(unique_tokens))}
        
        # Calculate IDF
        self.n_documents = len(corpus)
        document_frequency = {token: 0 for token in self.vocabulary_}
        
        for doc in corpus:
            tokens = set(self.tokenizer.tokenize(doc))
            for token in tokens:
                if token in document_frequency:
                    document_frequency[token] += 1
        
        # IDF = log(N / df) where N is total documents and df is document frequency
        for token, df in document_frequency.items():
            self.idf_[token] = math.log((self.n_documents + 1) / (df + 1)) + 1
    
    def transform(self, documents: List[str]) -> List[List[float]]:
        """
        Transform documents to TF-IDF feature vectors.
        
        Args:
            documents: List of text documents
            
        Returns:
            List of TF-IDF vectors
        """
        vectors = []
        vocab_size = len(self.vocabulary_)
        
        for doc in documents:
            # Initialize vector with zeros
            vector = [0.0] * vocab_size
            
            # Tokenize document
            tokens = self.tokenizer.tokenize(doc)
            
            # Calculate term frequency
            tf = {}
            for token in tokens:
                if token in self.vocabulary_:
                    tf[token] = tf.get(token, 0) + 1
            
            # Calculate TF-IDF
            total_terms = len(tokens)
            if total_terms > 0:
                for token, count in tf.items():
                    if token in self.vocabulary_ and token in self.idf_:
                        idx = self.vocabulary_[token]
                        # TF-IDF = (count / total_terms) * IDF
                        vector[idx] = (count / total_terms) * self.idf_[token]
            
            vectors.append(vector)
        
        return vectors
    
    def fit_transform(self, corpus: List[str]) -> List[List[float]]:
        """
        Learn vocabulary and IDF, then transform corpus to TF-IDF vectors.
        
        Args:
            corpus: List of text documents
            
        Returns:
            List of TF-IDF vectors
        """
        self.fit(corpus)
        return self.transform(corpus)
