package com.harito.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Tokenizer, Word2Vec}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import java.io.{File, PrintWriter}

object Lab17_NLPPipeline {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("NLP Pipeline Example")
      .master("local[*]")
      .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
      .getOrCreate()

    import spark.implicits._
    println("Spark Session created successfully.")
    println(s"Spark UI available at http://localhost:4040")
    println("Pausing for 10 seconds to allow you to open the Spark UI...")
    Thread.sleep(10000)
    
    // 1. --- Read Dataset ---
    val dataPath = "data/c4-train.00000-of-01024-30K.json.gz"
    val initialDF = spark.read.json(dataPath).limit(1000) // Limit for faster processing during lab
    println(s"Successfully read ${initialDF.count()} records.")
    initialDF.printSchema()
    println("\nSample of initial DataFrame:")
    initialDF.show(5, truncate = false) // Show full content for better understanding

    // --- Pipeline Stages Definition ---

    // 2. --- Tokenization ---
    // EXERCISE 1: Comment out the RegexTokenizer and uncomment the basic Tokenizer
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setPattern("\\s+|[.,;!?()\"']") // Fix: Use \\s for regex, and \" for double quote

    /*
    // Alternative Tokenizer: A simpler, whitespace-based tokenizer.
    // EXERCISE 1: Uncomment this for Exercise 1
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")
    */

    // 3. --- Stop Words Removal ---
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_tokens")

    // 4. --- Vectorization (Term Frequency) ---
    // EXERCISE 2: Change numFeatures from 20000 to 1000 to see the effect
    val hashingTF = new HashingTF()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("raw_features")
      .setNumFeatures(20000) // EXERCISE 2: Change this to 1000

    // 5. --- Vectorization (Inverse Document Frequency) ---
    val idf = new IDF()
      .setInputCol(hashingTF.getOutputCol)
      .setOutputCol("features")

    /*
    // EXERCISE 4: Comment out the HashingTF and IDF stages above, and uncomment Word2Vec below
    val word2Vec = new Word2Vec()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("word2vec_features")
      .setVectorSize(300)
      .setMinCount(2)
      .setWindowSize(5)
      .setMaxIter(10)
    */

    /*
    // EXERCISE 3: Uncomment this section to add LogisticRegression for classification
    // Add synthetic labels for classification (0 or 1 based on text length)
    val dataWithLabels = initialDF.withColumn("label", 
      when(length($"text") > 500, 1.0).otherwise(0.0))
    
    val logisticRegression = new LogisticRegression()
      .setFeaturesCol("features") // Use "word2vec_features" if using Word2Vec in Exercise 4
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setProbabilityCol("probability")
      .setMaxIter(20)
      .setRegParam(0.01)
    */

    // 6. --- Assemble the Pipeline ---
    // Default pipeline (Exercises 1 & 2)
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf))
    
    /*
    // EXERCISE 3: Use this pipeline instead for classification
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf, logisticRegression))
    */
    
    /*
    // EXERCISE 4: Use this pipeline for Word2Vec
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, word2Vec))
    */

    // 7. --- Fit the Pipeline ---
    println("\nFitting the NLP pipeline...")
    val fitStartTime = System.nanoTime()
    val pipelineModel = pipeline.fit(initialDF)
    val fitDuration = (System.nanoTime() - fitStartTime) / 1e9d
    println(f"--> Pipeline fitting took $fitDuration%.2f seconds.")
    
    /*
    // EXERCISE 3: Use dataWithLabels instead for classification
    val pipelineModel = pipeline.fit(dataWithLabels)
    */

    // 8. --- Transform the Data ---
    println("\nTransforming data with the fitted pipeline...")
    val transformStartTime = System.nanoTime()
    val transformedDF = pipelineModel.transform(initialDF)
    transformedDF.cache()
    val transformCount = transformedDF.count()
    val transformDuration = (System.nanoTime() - transformStartTime) / 1e9d
    println(f"--> Data transformation of $transformCount records took $transformDuration%.2f seconds.")
    
    /*
    // EXERCISE 3: Use dataWithLabels instead for classification
    val transformedDF = pipelineModel.transform(dataWithLabels)
    */

    // 9. --- Display Results ---
    println("\n" + "="*50)
    println("PIPELINE RESULTS")
    println("="*50)
    
    // Default display (Exercises 1 & 2)
    transformedDF.select($"text", $"tokens", $"filtered_tokens", $"features")
      .show(5, truncate = false)
    
    /*
    // EXERCISE 3: Uncomment to display classification results
    transformedDF.select($"text", $"features", $"label", $"prediction", $"probability")
      .show(5, truncate = false)
    */
    
    /*
    // EXERCISE 4: Uncomment to display Word2Vec results
    transformedDF.select($"text", $"word2vec_features")
      .show(5, truncate = false)
    */

    // Cleanup
    spark.stop()
    println("\nNLP Pipeline completed successfully!")
    println("To run different exercises, uncomment the appropriate sections as instructed above.")
  }
}
