from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.sql.functions import col
import os


def main():
    print("=" * 70)
    print("LAB 5: SENTIMENT ANALYSIS WITH PYSPARK")
    print("=" * 70)
    
    # Initialize Spark Session
    print("\n[Step 1] Initializing Spark Session")
    print("-" * 70)
    
    spark = SparkSession.builder \
        .appName("SentimentAnalysis") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    print("Spark Session created successfully")
    print(f"  Spark Version: {spark.version}")
    
    # Load Data
    print("\n[Step 2] Loading Data")
    print("-" * 70)
    
    data_path = "../data/sentiments.csv"
    
    # Check if file exists, if not create sample data
    if not os.path.exists(data_path):
        print("Creating sample sentiment data...")
        create_sample_data(spark, data_path)
    
    try:
        df = spark.read.csv(data_path, header=True, inferSchema=True)
        
        # Convert sentiment labels: -1/1 to 0/1
        df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
        
        # Drop rows with null sentiment values
        initial_row_count = df.count()
        df = df.dropna(subset=["sentiment"])
        final_row_count = df.count()
        
        print(f"Data loaded successfully")
        print(f"  Total rows: {initial_row_count}")
        print(f"  Rows after cleaning: {final_row_count}")
        print(f"  Dropped rows: {initial_row_count - final_row_count}")
        
        # Show sample data
        print("\nSample data:")
        df.select("text", "sentiment", "label").show(5, truncate=50)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating sample data instead...")
        df = create_sample_dataframe(spark)
    
    # Split data into training and testing sets
    print("\n[Step 3] Splitting Data")
    print("-" * 70)
    
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    
    print(f"Training samples: {train_data.count()}")
    print(f"Testing samples: {test_data.count()}")
    
    # Build Preprocessing Pipeline
    print("\n[Step 4] Building ML Pipeline")
    print("-" * 70)
    
    # Stage 1: Tokenizer - splits text into words
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    print("Tokenizer added")
    
    # Stage 2: StopWordsRemover - removes common words
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    print("StopWordsRemover added")
    
    # Stage 3: HashingTF - converts tokens to feature vectors
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    print("HashingTF added (features: 10000)")
    
    # Stage 4: IDF - inverse document frequency
    idf = IDF(inputCol="raw_features", outputCol="features")
    print("IDF added")
    
    # Stage 5: Logistic Regression - the classifier
    lr = LogisticRegression(
        maxIter=10,
        regParam=0.001,
        featuresCol="features",
        labelCol="label"
    )
    print("LogisticRegression added")
    
    # Assemble the pipeline
    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf, lr])
    print("\nPipeline assembled with 5 stages")
    
    # Train the model
    print("\n[Step 5] Training the Model")
    print("-" * 70)
    
    print("Training in progress...")
    model = pipeline.fit(train_data)
    print("Model trained successfully")
    
    # Make predictions
    print("\n[Step 6] Making Predictions")
    print("-" * 70)
    
    predictions = model.transform(test_data)
    print("Predictions generated")
    
    # Show some predictions
    print("\nSample predictions:")
    predictions.select("text", "label", "prediction", "probability").show(5, truncate=50)
    
    # Evaluate the model
    print("\n[Step 7] Model Evaluation")
    print("-" * 70)
    
    # Accuracy
    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = accuracy_evaluator.evaluate(predictions)
    
    # F1 Score
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
    f1 = f1_evaluator.evaluate(predictions)
    
    # AUC (Area Under ROC Curve)
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc = auc_evaluator.evaluate(predictions)
    
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    predictions.groupBy("label", "prediction").count().show()
    
    # Test on custom examples
    print("\n[Step 8] Testing on Custom Examples")
    print("-" * 70)
    
    test_texts = [
        "This is an amazing movie, I loved it!",
        "Terrible film, complete waste of time.",
        "Great acting and wonderful story.",
        "Boring and disappointing.",
    ]
    
    test_df = spark.createDataFrame(
        [(text,) for text in test_texts],
        ["text"]
    )
    
    custom_predictions = model.transform(test_df)
    
    print("\nCustom Predictions:")
    custom_predictions.select("text", "prediction", "probability").show(truncate=False)
    
    # Stop Spark Session
    print("\n" + "=" * 70)
    print("Analysis completed successfully!")
    print("=" * 70)
    
    spark.stop()


def create_sample_data(spark, output_path):
    """Create sample sentiment data for demonstration"""
    sample_data = [
        ("This movie is fantastic and I love it!", 1),
        ("I hate this film, it's terrible.", -1),
        ("The acting was superb, a truly great experience.", 1),
        ("What a waste of time, absolutely boring.", -1),
        ("Highly recommend this, a masterpiece.", 1),
        ("Could not finish watching, so bad.", -1),
        ("Amazing performance, brilliant story!", 1),
        ("Awful movie, complete disaster.", -1),
        ("Best film I've seen this year!", 1),
        ("Terrible acting and boring plot.", -1),
        ("Absolutely wonderful, a must watch!", 1),
        ("Disappointing and dull movie.", -1),
        ("Excellent cinematography and direction.", 1),
        ("Poor script and bad editing.", -1),
        ("Captivating from start to finish.", 1),
        ("Could not connect with any character.", -1),
        ("A true masterpiece of cinema!", 1),
        ("Overhyped and underwhelming.", -1),
        ("Beautifully crafted story.", 1),
        ("Confusing and poorly executed.", -1),
    ]
    
    df = spark.createDataFrame(sample_data, ["text", "sentiment"])
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.toPandas().to_csv(output_path, index=False)
    print(f"Sample data created at {output_path}")


def create_sample_dataframe(spark):
    """Create sample DataFrame directly"""
    sample_data = [
        ("This movie is fantastic and I love it!", 1, 1.0),
        ("I hate this film, it's terrible.", -1, 0.0),
        ("The acting was superb, a truly great experience.", 1, 1.0),
        ("What a waste of time, absolutely boring.", -1, 0.0),
        ("Highly recommend this, a masterpiece.", 1, 1.0),
        ("Could not finish watching, so bad.", -1, 0.0),
        ("Amazing performance, brilliant story!", 1, 1.0),
        ("Awful movie, complete disaster.", -1, 0.0),
        ("Best film I've seen this year!", 1, 1.0),
        ("Terrible acting and boring plot.", -1, 0.0),
        ("Absolutely wonderful, a must watch!", 1, 1.0),
        ("Disappointing and dull movie.", -1, 0.0),
        ("Excellent cinematography and direction.", 1, 1.0),
        ("Poor script and bad editing.", -1, 0.0),
        ("Captivating from start to finish.", 1, 1.0),
        ("Could not connect with any character.", -1, 0.0),
        ("A true masterpiece of cinema!", 1, 1.0),
        ("Overhyped and underwhelming.", -1, 0.0),
        ("Beautifully crafted story.", 1, 1.0),
        ("Confusing and poorly executed.", -1, 0.0),
    ]
    
    df = spark.createDataFrame(sample_data, ["text", "sentiment", "label"])
    return df


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
