package com.harito.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Tokenizer, Word2Vec, Normalizer}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row
import java.io.{File, PrintWriter}

object Lab17_NLPPipeline {
  def main(args: Array[String]): Unit = {
    // Biến để tùy chỉnh số lượng tài liệu
    val limitDocuments = 1000
    
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
    val initialDF = spark.read.json(dataPath).limit(limitDocuments) // Limit for faster processing during lab
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
      .setOutputCol("tf_idf_features")

    // 6. --- Normalization of Count Vectors ---
    val normalizer = new Normalizer()
      .setInputCol("tf_idf_features")
      .setOutputCol("normalized_features")
      .setP(2.0) // L2 normalization (Euclidean norm)

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

    // 7. --- Assemble the Pipeline ---
    // Default pipeline with normalization (Exercises 1 & 2)
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf, normalizer))
    
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

    // 8. --- Fit the Pipeline ---
    println("\nFitting the NLP pipeline...")
    val fitStartTime = System.nanoTime()
    val pipelineModel = pipeline.fit(initialDF)
    val fitDuration = (System.nanoTime() - fitStartTime) / 1e9d
    println(f"--> Pipeline fitting took $fitDuration%.2f seconds.")
    
    /*
    // EXERCISE 3: Use dataWithLabels instead for classification
    val pipelineModel = pipeline.fit(dataWithLabels)
    */

    // 9. --- Transform the Data ---
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

    // 10. --- Display Results ---
    println("\n" + "="*50)
    println("PIPELINE RESULTS")
    println("="*50)
    
    // Default display with normalized features (Exercises 1 & 2)
    transformedDF.select($"text", $"tokens", $"filtered_tokens", $"tf_idf_features", $"normalized_features")
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

    // 11. --- Cosine Similarity Demo ---
    println("\n" + "="*50)
    println("COSINE SIMILARITY SEARCH DEMO")
    println("="*50)
    
    // Function to calculate cosine similarity between two vectors
    def cosineSimilarity(v1: Vector, v2: Vector): Double = {
      val dot = v1.dot(v2)
      val norm1 = math.sqrt(v1.dot(v1))
      val norm2 = math.sqrt(v2.dot(v2))
      if (norm1 == 0.0 || norm2 == 0.0) 0.0 else dot / (norm1 * norm2)
    }
    
    // Collect all vectors with their indices and text
    val vectorsWithText = transformedDF
      .select($"text", $"normalized_features")
      .collect()
      .zipWithIndex
      .map { case (row, idx) => 
        (idx, row.getAs[String]("text"), row.getAs[Vector]("normalized_features"))
      }
    
    // Choose a random document (let's pick index 3)
    val queryIndex = 3
    val (_, queryText, queryVector) = vectorsWithText(queryIndex)
    
    println(s"Query Document (Index $queryIndex):")
    println(s"Text: ${queryText.take(100)}...")
    println()
    
    // Calculate similarities with all other documents
    val similarities = vectorsWithText
      .filter(_._1 != queryIndex) // Exclude the query document itself
      .map { case (idx, text, vector) =>
        val similarity = cosineSimilarity(queryVector, vector)
        (idx, text, vector, similarity)
      }
      .sortBy(-_._4) // Sort by similarity descending
    
    // Show top 5 most similar documents
    println("TOP 5 MOST SIMILAR DOCUMENTS:")
    println("-" * 50)
    similarities.take(5).zipWithIndex.foreach { case ((idx, text, _, similarity), rank) =>
      println(f"${rank + 1}. Document $idx (Similarity: $similarity%.4f)")
      println(f"   Text: ${text.take(80)}...")
      println()
    }
    
    // Show the most similar document in detail
    if (similarities.nonEmpty) {
      val (mostSimilarIdx, mostSimilarText, _, maxSimilarity) = similarities.head
      println("="*50)
      println("MOST SIMILAR DOCUMENT DETAILS:")
      println("="*50)
      println(f"Document Index: $mostSimilarIdx")
      println(f"Cosine Similarity: $maxSimilarity%.6f")
      println(f"Query Text: $queryText")
      println(f"\nMost Similar Text: $mostSimilarText")
      println("="*50)
    }

    // Cleanup
    spark.stop()
    println("\nNLP Pipeline with Cosine Similarity Search completed successfully!")
    println("To run different exercises, uncomment the appropriate sections as instructed above.")
  }
}
