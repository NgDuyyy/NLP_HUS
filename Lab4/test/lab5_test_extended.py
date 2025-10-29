import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.tokenizer import RegexTokenizer
from src.models.tfidf_vectorizer import TfidfVectorizer
from src.models.text_classifier import TextClassifier
from sklearn.model_selection import train_test_split


def main():
    print("=" * 70)
    print("LAB 5: TEXT CLASSIFICATION - EXTENDED DATASET")
    print("=" * 70)
    
    # Extended Dataset with more samples
    texts = [
        # Positive reviews
        "This movie is fantastic and I love it!",
        "The acting was superb, a truly great experience.",
        "Highly recommend this, a masterpiece.",
        "Amazing performance, brilliant story!",
        "Best film I've seen this year!",
        "Absolutely wonderful, a must watch!",
        "Excellent cinematography and direction.",
        "Captivating from start to finish.",
        "A true masterpiece of cinema!",
        "Beautifully crafted story.",
        "Outstanding performances by all actors.",
        "Incredible visuals and soundtrack.",
        "A work of art, simply stunning.",
        "Deeply moving and powerful.",
        "Brilliantly written and executed.",
        "Exceeded all my expectations.",
        "One of the best movies ever made.",
        "Absolutely loved every minute.",
        "Phenomenal acting and directing.",
        "A must-see for everyone.",
        "Perfect from beginning to end.",
        "Remarkable storytelling.",
        "Exceptional in every way.",
        "Truly inspiring and uplifting.",
        "A cinematic triumph.",
        
        # Negative reviews
        "I hate this film, it's terrible.",
        "What a waste of time, absolutely boring.",
        "Could not finish watching, so bad.",
        "Awful movie, complete disaster.",
        "Terrible acting and boring plot.",
        "Disappointing and dull movie.",
        "Poor script and bad editing.",
        "Could not connect with any character.",
        "Overhyped and underwhelming.",
        "Confusing and poorly executed.",
        "Complete waste of money.",
        "Boring from start to finish.",
        "Poorly written and badly acted.",
        "Utterly disappointing experience.",
        "Failed to engage the audience.",
        "Predictable and unoriginal.",
        "Lacked depth and substance.",
        "Tedious and uninteresting.",
        "Not worth watching at all.",
        "A total mess of a film.",
        "Painfully slow and boring.",
        "Worst movie I've seen.",
        "Lacking any redeeming qualities.",
        "Frustrating and annoying.",
        "An absolute waste of time.",
    ]
    
    # Labels: 1 for positive (first 25), 0 for negative (last 25)
    labels = [1] * 25 + [0] * 25
    
    print(f"\n[Data Information]")
    print("-" * 70)
    print(f"Total samples: {len(texts)}")
    print(f"Positive samples: {sum(labels)}")
    print(f"Negative samples: {len(labels) - sum(labels)}")
    
    # Split data (70% train, 30% test for better evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Initialize components
    print("\n[Initialize Components]")
    print("-" * 70)
    
    tokenizer = RegexTokenizer(lowercase=True)
    vectorizer = TfidfVectorizer(tokenizer)
    classifier = TextClassifier(vectorizer)
    
    print("All components initialized")
    
    # Train the classifier
    print("\n[Training]")
    print("-" * 70)
    
    classifier.fit(X_train, y_train)
    
    model_info = classifier.get_model_info()
    print(f"\nVocabulary size: {model_info['vocabulary_size']}")
    print(f"Classes: {model_info['classes']}")
    
    # Make predictions on training set
    print("\n[Training Set Performance]")
    print("-" * 70)
    
    y_train_pred = classifier.predict(X_train)
    train_metrics = classifier.evaluate(y_train, y_train_pred)
    
    print("Training Metrics:")
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f} ({train_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall:    {train_metrics['recall']:.4f}")
    print(f"  F1-Score:  {train_metrics['f1_score']:.4f}")
    
    # Make predictions on test set
    print("\n[Testing Set Performance]")
    print("-" * 70)
    
    y_test_pred = classifier.predict(X_test)
    test_metrics = classifier.evaluate(y_test, y_test_pred)
    
    print("Testing Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    
    # Show some predictions
    print("\n[Sample Predictions]")
    print("-" * 70)
    
    correct = 0
    incorrect = 0
    
    for i, (text, true_label, pred_label) in enumerate(zip(X_test, y_test, y_test_pred)):
        if true_label == pred_label:
            correct += 1
        else:
            incorrect += 1
            sentiment_true = "Positive" if true_label == 1 else "Negative"
            sentiment_pred = "Positive" if pred_label == 1 else "Negative"
            
            print(f"\nMisclassified:")
            print(f"  Text: \"{text}\"")
            print(f"  True: {sentiment_true} | Predicted: {sentiment_pred}")

    print(f"\nCorrectly classified: {correct}/{len(y_test)}")
    print(f"Misclassified: {incorrect}/{len(y_test)}")
    
    # Test on completely new examples
    print("\n[Testing on New Examples]")
    print("-" * 70)
    
    new_texts = [
        "This is an incredible film with outstanding performances!",
        "Very disappointing and not worth the money.",
        "Absolutely loved this movie, highly recommend!",
        "Boring plot and terrible acting.",
        "A perfect blend of drama and action!",
        "Waste of time, extremely dull.",
        "Masterfully directed with brilliant cinematography!",
        "Could not understand the plot, very confusing.",
    ]
    
    new_predictions = classifier.predict(new_texts)
    new_probabilities = classifier.predict_proba(new_texts)
    
    for i, (text, pred, prob) in enumerate(zip(new_texts, new_predictions, new_probabilities)):
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = max(prob) * 100
        
        print(f"\n{i+1}. \"{text}\"")
        print(f"   → {sentiment} (Confidence: {confidence:.2f}%)")
        print(f"   Probabilities: [Negative: {prob[0]:.3f}, Positive: {prob[1]:.3f}]")
    
    print("\n" + "=" * 70)
    print("Extended testing completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
