import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.tokenizer import RegexTokenizer
from src.models.tfidf_vectorizer import TfidfVectorizer
from src.models.text_classifier import TextClassifier


def main():
    print("=" * 70)
    print("LAB 5: TEXT CLASSIFICATION - QUICK DEMO")
    print("=" * 70)
    
    # Sample training data
    train_texts = [
        "This movie is fantastic and amazing!",
        "I love this film, it's wonderful!",
        "Great acting and superb story!",
        "Excellent movie, highly recommend!",
        "Best film I've ever seen!",
        
        "This movie is terrible and boring.",
        "I hate this film, complete waste.",
        "Awful acting and terrible plot.",
        "Worst movie ever, don't watch.",
        "Disappointing and very bad.",
    ]
    
    train_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    
    print("\n[Step 1] Training the classifier...")
    print("-" * 70)
    
    # Initialize and train
    tokenizer = RegexTokenizer(lowercase=True)
    vectorizer = TfidfVectorizer(tokenizer)
    classifier = TextClassifier(vectorizer)
    
    classifier.fit(train_texts, train_labels)
    print("Model trained successfully!")
    
    # Interactive demo
    print("\n[Step 2] Interactive Testing")
    print("-" * 70)
    print("\nTry some test sentences:\n")
    
    test_examples = [
        "This is an amazing and wonderful movie!",
        "Terrible film, absolutely horrible.",
        "Great performance and excellent story!",
        "Boring and disappointing waste of time.",
    ]
    
    for i, text in enumerate(test_examples, 1):
        prediction = classifier.predict([text])[0]
        proba = classifier.predict_proba([text])[0]
        
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        confidence = max(proba) * 100
        
        print(f"{i}. \"{text}\"")
        print(f"   → {sentiment} (Confidence: {confidence:.1f}%)")
        print(f"   Probabilities: [Neg: {proba[0]:.3f}, Pos: {proba[1]:.3f}]")
        print()
    
    # User input mode
    print("-" * 70)
    print("\n[Step 3] Try your own text!")
    print("-" * 70)
    print("Type your review (or 'quit' to exit):\n")
    
    while True:
        try:
            user_input = input(">>> ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the demo!")
                break
            
            if not user_input.strip():
                continue
            
            prediction = classifier.predict([user_input])[0]
            proba = classifier.predict_proba([user_input])[0]

            sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
            confidence = max(proba) * 100
            
            print(f"\n   Prediction: {sentiment}")
            print(f"   Confidence: {confidence:.1f}%")
            print(f"   Probabilities: [Negative: {proba[0]:.3f}, Positive: {proba[1]:.3f}]")
            print()
            
        except KeyboardInterrupt:
            print("\n\nDemo interrupted.")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
