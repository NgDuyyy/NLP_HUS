import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.tokenizer import RegexTokenizer
from src.models.tfidf_vectorizer import TfidfVectorizer
from src.models.text_classifier import TextClassifier
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import time


def get_balanced_subset(texts, labels, n_samples):
    """Get a balanced subset of data with equal representation of both classes"""
    # Separate texts by label
    positive_texts = [text for text, label in zip(texts, labels) if label == 1]
    negative_texts = [text for text, label in zip(texts, labels) if label == 0]
    
    # Calculate samples per class
    samples_per_class = n_samples // 2
    
    # Get subset
    subset_positive = positive_texts[:samples_per_class]
    subset_negative = negative_texts[:samples_per_class]
    
    # Combine and shuffle
    subset_texts = subset_positive + subset_negative
    subset_labels = [1] * len(subset_positive) + [0] * len(subset_negative)
    
    # Shuffle together
    combined = list(zip(subset_texts, subset_labels))
    import random
    random.seed(42)
    random.shuffle(combined)
    subset_texts, subset_labels = zip(*combined)
    
    return list(subset_texts), list(subset_labels)


def main():
    print("=" * 80)
    print("LAB 5: TEXT CLASSIFICATION - HUGGING FACE DATASET")
    print("Dataset: Twitter Financial News Sentiment")
    print("=" * 80)
    
    # Load dataset from Hugging Face
    print("\n[Step 1] Loading Dataset from Hugging Face")
    print("-" * 80)
    print("Downloading and loading dataset... (this may take a moment)")
    
    try:
        ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
        print("Dataset loaded successfully")
        
        # Explore dataset structure
        print(f"\nDataset structure: {ds}")
        print(f"Available splits: {list(ds.keys())}")
        
        # Use validation split if available, otherwise train split
        if 'validation' in ds:
            dataset = ds['validation']
        else:
            dataset = ds['train']
        
        print(f"\nUsing split: {list(ds.keys())[0] if 'train' in ds else 'validation'}")
        print(f"Total samples: {len(dataset)}")
        
        # Show sample data
        print("\nSample data:")
        for i in range(min(3, len(dataset))):
            print(f"  {i+1}. Text: {dataset[i]['text'][:80]}...")
            print(f"     Label: {dataset[i]['label']}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nFalling back to local extended dataset...")
        return use_extended_dataset()
    
    # Prepare data
    print("\n[Step 2] Preparing Data")
    print("-" * 80)
    
    # Extract texts and labels
    texts = [item['text'] for item in dataset]
    labels = [item['label'] for item in dataset]
    
    # Check label distribution
    unique_labels = set(labels)
    print(f"Unique labels: {sorted(unique_labels)}")
    
    label_counts = {}
    for label in unique_labels:
        label_counts[label] = labels.count(label)
    
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        percentage = (count / len(labels)) * 100
        print(f"  Label {label}: {count} samples ({percentage:.2f}%)")
    
    # For binary classification, we'll convert labels if needed
    # Assuming labels are: 0 (negative), 1 (neutral), 2 (positive)
    # Convert to binary: 0 (negative/neutral), 1 (positive)
    if len(unique_labels) > 2:
        print("\nConverting to binary classification (Positive vs Non-Positive)...")
        # Map: 0,1 -> 0 (non-positive), 2 -> 1 (positive)
        labels = [1 if label == 2 else 0 for label in labels]
        print("Labels converted")
    
    # Limit dataset size for faster training (optional)
    max_samples = 2000  # Adjust this value based on your needs
    if len(texts) > max_samples:
        print(f"\nLimiting dataset to {max_samples} samples for faster training...")
        texts = texts[:max_samples]
        labels = labels[:max_samples]
    
    print(f"\nFinal dataset size: {len(texts)}")
    print(f"Positive samples: {sum(labels)}")
    print(f"Negative samples: {len(labels) - sum(labels)}")
    
    # Compare with small dataset first
    # Ensure balanced samples for small experiments
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: SMALL DATASET (100 samples)")
    print("=" * 80)
    # Get balanced subset
    small_texts, small_labels = get_balanced_subset(texts, labels, 100)
    run_experiment(small_texts, small_labels, "Small Dataset")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: MEDIUM DATASET (500 samples)")
    print("=" * 80)
    medium_texts, medium_labels = get_balanced_subset(texts, labels, 500)
    run_experiment(medium_texts, medium_labels, "Medium Dataset")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: LARGE DATASET (All samples)")
    print("=" * 80)
    run_experiment(texts, labels, "Large Dataset")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & CONCLUSIONS")
    print("=" * 80)
    print("""
Key Observations:
1. More data generally leads to better generalization
2. High-quality labeled data improves model accuracy
3. Larger vocabulary provides more features for classification
4. Training time increases with dataset size
5. Test accuracy is the key metric for real-world performance

Recommendations:
- Use as much high-quality labeled data as possible
- Balance the dataset to avoid bias
- Consider data augmentation for small datasets
- Monitor both training and testing metrics to detect overfitting
    """)


def run_experiment(texts, labels, experiment_name):
    """Run a complete experiment with given data"""
    
    print(f"\n[{experiment_name}]")
    print("-" * 80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Initialize components
    tokenizer = RegexTokenizer(lowercase=True)
    vectorizer = TfidfVectorizer(tokenizer)
    classifier = TextClassifier(vectorizer)
    
    # Train
    print("\nTraining...")
    start_time = time.time()
    classifier.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    model_info = classifier.get_model_info()
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"  Vocabulary size: {model_info['vocabulary_size']}")
    
    # Evaluate on training set
    y_train_pred = classifier.predict(X_train)
    train_metrics = classifier.evaluate(y_train, y_train_pred)
    
    # Evaluate on test set
    y_test_pred = classifier.predict(X_test)
    test_metrics = classifier.evaluate(y_test, y_test_pred)
    
    # Display results
    print("\n" + "-" * 40)
    print("TRAINING SET PERFORMANCE:")
    print("-" * 40)
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f} ({train_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall:    {train_metrics['recall']:.4f}")
    print(f"  F1-Score:  {train_metrics['f1_score']:.4f}")
    
    print("\n" + "-" * 40)
    print("TESTING SET PERFORMANCE:")
    print("-" * 40)
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    
    # Calculate overfitting indicator
    overfit_gap = train_metrics['accuracy'] - test_metrics['accuracy']
    print("\n" + "-" * 40)
    print("ANALYSIS:")
    print("-" * 40)
    print(f"  Overfitting gap: {overfit_gap:.4f} ({overfit_gap*100:.2f}%)")
    if overfit_gap > 0.10:
        print("  Warning: Model may be overfitting (gap > 10%)")
    elif overfit_gap > 0.05:
        print("  Caution: Slight overfitting detected (gap 5-10%)")
    else:
        print("  Good generalization (gap < 5%)")

    print(f"  Training time: {training_time:.2f}s")
    print(f"  Samples per second: {len(X_train)/training_time:.0f}")
    
    # Show some misclassifications
    misclassified = []
    for text, true_label, pred_label in zip(X_test, y_test, y_test_pred):
        if true_label != pred_label:
            misclassified.append((text, true_label, pred_label))
    
    if misclassified:
        print(f"\n  Misclassified: {len(misclassified)}/{len(y_test)} samples")
        print("\n  Sample misclassifications:")
        for i, (text, true_label, pred_label) in enumerate(misclassified[:3]):
            sentiment_true = "Positive" if true_label == 1 else "Negative"
            sentiment_pred = "Positive" if pred_label == 1 else "Negative"
            print(f"\n  {i+1}. \"{text[:80]}...\"")
            print(f"     True: {sentiment_true} | Predicted: {sentiment_pred}")
    
    return test_metrics


def use_extended_dataset():
    """Fallback to extended local dataset"""
    print("Using extended local dataset...")
    
    texts = [
        # Positive reviews (25)
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
        # Negative reviews (25)
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
    
    labels = [1] * 25 + [0] * 25
    
    run_experiment(texts, labels, "Extended Local Dataset")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
