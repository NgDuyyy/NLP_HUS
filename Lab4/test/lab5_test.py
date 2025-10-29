import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.tokenizer import RegexTokenizer
from src.models.tfidf_vectorizer import TfidfVectorizer
from src.models.text_classifier import TextClassifier
from sklearn.model_selection import train_test_split


def main():
    print("=" * 70)
    print("LAB 5: TEXT CLASSIFICATION")
    print("=" * 70)
    
    # Task 1: Data Preparation
    print("\n[Task 1] Data Preparation")
    print("-" * 70)
    
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad.",
        "Amazing performance, brilliant story!",
        "Awful movie, complete disaster.",
        "Best film I've seen this year!",
        "Terrible acting and boring plot.",
        "Absolutely wonderful, a must watch!",
        "Disappointing and dull movie.",
    ]
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative
    
    print(f"Total samples: {len(texts)}")
    print(f"Positive samples: {sum(labels)}")
    print(f"Negative samples: {len(labels) - sum(labels)}")
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Task 2: Initialize components
    print("\n[Task 2] Initialize Tokenizer and Vectorizer")
    print("-" * 70)
    
    # Initialize RegexTokenizer
    tokenizer = RegexTokenizer(lowercase=True)
    print("RegexTokenizer initialized")
    
    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer(tokenizer)
    print("TfidfVectorizer initialized")
    
    # Initialize TextClassifier
    classifier = TextClassifier(vectorizer)
    print("TextClassifier initialized")

    # Task 3: Train the classifier
    print("\n[Task 3] Training the Classifier")
    print("-" * 70)
    
    classifier.fit(X_train, y_train)
    print("Training completed")
    
    # Display model information
    model_info = classifier.get_model_info()
    print(f"\nModel Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Task 4: Make predictions
    print("\n[Task 4] Making Predictions on Test Set")
    print("-" * 70)
    
    y_pred = classifier.predict(X_test)
    
    print("\nTest Results:")
    for i, (text, true_label, pred_label) in enumerate(zip(X_test, y_test, y_pred)):
        sentiment_true = "Positive" if true_label == 1 else "Negative"
        sentiment_pred = "Positive" if pred_label == 1 else "Negative"
        status = "pos" if true_label == pred_label else "neg"
        
        print(f"\n{status} Sample {i+1}:")
        print(f"  Text: \"{text}\"")
        print(f"  True: {sentiment_true} | Predicted: {sentiment_pred}")
    
    # Task 5: Evaluate the model
    print("\n[Task 5] Model Evaluation")
    print("-" * 70)
    
    metrics = classifier.evaluate(y_test, y_pred)
    
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    # Additional: Test on new examples
    print("\n[Bonus] Testing on New Examples")
    print("-" * 70)
    
    new_texts = [
        "This is an excellent movie with great acting!",
        "Boring and disappointing film.",
        "I really enjoyed watching this masterpiece.",
        "Waste of money, very bad movie.",
    ]
    
    new_predictions = classifier.predict(new_texts)
    new_probabilities = classifier.predict_proba(new_texts)
    
    print("\nNew Predictions:")
    for i, (text, pred, prob) in enumerate(zip(new_texts, new_predictions, new_probabilities)):
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = max(prob) * 100
        
        print(f"\n  {i+1}. \"{text}\"")
        print(f"     Prediction: {sentiment}")
        print(f"     Confidence: {confidence:.2f}%")
        print(f"     Probabilities: [Neg: {prob[0]:.3f}, Pos: {prob[1]:.3f}]")
    
    print("\n" + "=" * 70)
    print("Testing completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
