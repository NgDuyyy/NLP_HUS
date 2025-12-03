from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TextClassifier:
    """
    A text classifier that uses a vectorizer and logistic regression 
    to classify text documents.
    """
    
    def __init__(self, vectorizer):
        """
        Initialize the TextClassifier.
        
        Args:
            vectorizer: A vectorizer instance (e.g., TfidfVectorizer, CountVectorizer)
                       that implements fit_transform() and transform() methods
        """
        self.vectorizer = vectorizer
        self._model = None
    
    def fit(self, texts: List[str], labels: List[int]):
        """
        Train the classifier on the provided texts and labels.
        
        Args:
            texts: List of text documents
            labels: List of corresponding labels (e.g., 0 for negative, 1 for positive)
        """
        # Vectorize the training texts
        X = self.vectorizer.fit_transform(texts)
        
        # Initialize and train the logistic regression model
        self._model = LogisticRegression(solver='liblinear', random_state=42)
        self._model.fit(X, labels)
        
        print(f"Model trained on {len(texts)} samples")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def predict(self, texts: List[str]) -> List[int]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of text documents to classify
            
        Returns:
            List of predicted labels
        """
        if self._model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Vectorize the input texts
        X = self.vectorizer.transform(texts)
        
        # Make predictions
        predictions = self._model.predict(X)
        
        return predictions.tolist()
    
    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        """
        Predict class probabilities for texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of probability arrays for each class
        """
        if self._model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        X = self.vectorizer.transform(texts)
        probabilities = self._model.predict_proba(X)
        
        return probabilities.tolist()
    
    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Calculate evaluation metrics for the predictions.
        
        Args:
            y_true: List of true labels
            y_pred: List of predicted labels
            
        Returns:
            Dictionary containing accuracy, precision, recall, and f1-score
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        return metrics
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary containing model information
        """
        if self._model is None:
            return {"status": "Model not trained"}
        
        return {
            "model_type": "Logistic Regression",
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "classes": self._model.classes_.tolist(),
            "coefficients_shape": self._model.coef_.shape
        }
