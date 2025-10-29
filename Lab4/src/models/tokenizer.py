import re
from typing import List


class RegexTokenizer:
    """
    A tokenizer that uses regular expressions to split text into tokens.
    It extracts words (alphanumeric sequences) and punctuation marks.
    """
    
    def __init__(self, lowercase: bool = True):
        """
        Initialize the RegexTokenizer.
        
        Args:
            lowercase: Whether to convert text to lowercase before tokenization
        """
        self.lowercase = lowercase
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text using regex pattern.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        if self.lowercase:
            text = text.lower()
        
        # Pattern matches words (alphanumeric) or individual punctuation marks
        tokens = re.findall(r"\w+|[^\w\s]", text)
        
        return tokens


class SimpleTokenizer:
    """
    A simple tokenizer that splits text by whitespace and removes punctuation.
    """
    
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        """
        Initialize the SimpleTokenizer.
        
        Args:
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text by splitting on whitespace.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            # Remove punctuation
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split by whitespace and filter empty strings
        tokens = [token for token in text.split() if token]
        
        return tokens
