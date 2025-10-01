package com.harito.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import java.io.{File, PrintWriter, FileWriter, BufferedWriter}
import java.time.{LocalDateTime, ZoneId}
import java.time.format.DateTimeFormatter
import scala.util.{Try, Success, Failure}

/**
 * NLP Data Pipeline with Spark
 * 
 * Requirements:
 * 1. Read compressed JSON data (c4-train.00000-of-01024-30K.json.gz)
 * 2. Perform text preprocessing (tokenization, stop word removal)
 * 3. Vectorize data using HashingTF and IDF
 * 4. Save results to results/lab17_pipeline_output.txt
 * 5. Log the entire process to log/ directory
 */
object Lab17_DataPipeline {
  
  // Setup directories and logging
  val resultsDir = new File("results")
  val logDir = new File("log")
  if (!resultsDir.exists()) resultsDir.mkdirs()
  if (!logDir.exists()) logDir.mkdirs()
  
  val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
  val startTime = LocalDateTime.now()
  val logFile = new File(s"log/lab17_pipeline_${startTime.format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))}.log")
  val logWriter = new BufferedWriter(new FileWriter(logFile))
  
  /**
   * Logging function that writes both to console and log file
   */
  def log(message: String): Unit = {
    val timestamp = LocalDateTime.now().format(formatter)
    val logMessage = s"[$timestamp] $message"
    println(logMessage)
    logWriter.write(logMessage + "\n")
    logWriter.flush()
  }

  def main(args: Array[String]): Unit = {
    log("=== STARTING NLP DATA PIPELINE WITH SPARK ===")
    log(s"Job started at: ${startTime.format(formatter)}")
    
    val pipelineResult = Try {
      // ===============================================
      // STEP 1: Initialize Spark Session
      // ===============================================
      log("STEP 1: Initializing Spark Session")
      val spark = SparkSession.builder
        .appName("NLP Data Pipeline - Lab 17")
        .master("local[*]") // Use all available cores
        .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()

      import spark.implicits._
      
      log("Spark Session created successfully")
      log(s"Spark version: ${spark.version}")
      log(s"Spark UI available at: http://localhost:4040")
      log(s"Using ${spark.sparkContext.defaultParallelism} cores for processing")
      
      // ===============================================
      // STEP 2: Read the Data
      // ===============================================
      log("STEP 2: Reading compressed JSON data")
      val dataPath = "data/c4-train.00000-of-01024-30K.json.gz"
      log(s"Loading data from: $dataPath")
      
      val loadStartTime = System.nanoTime()
      val initialDF = spark.read
        .option("multiline", "false") // Each line is a separate JSON record
        .option("mode", "PERMISSIVE") // Handle malformed records gracefully
        .json(dataPath)
      
      val recordCount = initialDF.count()
      val loadDuration = (System.nanoTime() - loadStartTime) / 1e9d
      
      log(f"Successfully loaded $recordCount records in $loadDuration%.2f seconds")
      log("Data schema:")
      initialDF.printSchema()
      
      // Show sample of original data
      log("Sample of original data:")
      initialDF.show(3, truncate = true)
      
      // ===============================================
      // STEP 3: Text Preprocessing Pipeline
      // ===============================================
      log("STEP 3: Setting up Text Preprocessing Pipeline")
      
      // 3.1 Tokenization
      log("Configuring RegexTokenizer for text tokenization")
      val tokenizer = new RegexTokenizer()
        .setInputCol("text")
        .setOutputCol("tokens")
        .setPattern("\\s+|[.,;!?()\"']") // Split on whitespace and punctuation
        .setGaps(true) // Use gaps between matches as tokens
      
      // 3.2 Stop Words Removal
      log("Configuring StopWordsRemover to filter common words")
      val stopWordsRemover = new StopWordsRemover()
        .setInputCol(tokenizer.getOutputCol)
        .setOutputCol("filtered_tokens")
        .setCaseSensitive(false)
      
      // ===============================================
      // STEP 4: Vectorization
      // ===============================================
      log("STEP 4: Setting up Vectorization Pipeline")
      
      // 4.1 HashingTF (Term Frequency)
      val numFeatures = 20000 // Use 20K features for good performance
      log(s"Configuring HashingTF with $numFeatures features")
      val hashingTF = new HashingTF()
        .setInputCol(stopWordsRemover.getOutputCol)
        .setOutputCol("raw_features")
        .setNumFeatures(numFeatures)
      
      // 4.2 IDF (Inverse Document Frequency)
      log("Configuring IDF for inverse document frequency weighting")
      val idf = new IDF()
        .setInputCol(hashingTF.getOutputCol)
        .setOutputCol("features")
        .setMinDocFreq(2) // Ignore terms that appear in less than 2 documents
      
      // ===============================================
      // STEP 5: Assemble and Fit Pipeline
      // ===============================================
      log("STEP 5: Assembling and fitting the complete pipeline")
      val pipeline = new Pipeline()
        .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf))
      
      log(s"Pipeline assembled with ${pipeline.getStages.length} stages")
      log("Pipeline stages: Tokenizer -> StopWordsRemover -> HashingTF -> IDF")
      
      // Fit the pipeline
      val fitStartTime = System.nanoTime()
      log("Fitting pipeline to data...")
      val pipelineModel = pipeline.fit(initialDF)
      val fitDuration = (System.nanoTime() - fitStartTime) / 1e9d
      log(f"Pipeline fitting completed in $fitDuration%.2f seconds")
      
      // ===============================================
      // STEP 6: Transform Data
      // ===============================================
      log("STEP 6: Transforming data with fitted pipeline")
      val transformStartTime = System.nanoTime()
      val transformedDF = pipelineModel.transform(initialDF)
      
      // Cache the result for multiple operations
      transformedDF.cache()
      val finalRecordCount = transformedDF.count()
      val transformDuration = (System.nanoTime() - transformStartTime) / 1e9d
      
      log(f"Data transformation completed in $transformDuration%.2f seconds")
      log(s"Processed $finalRecordCount records successfully")
      
      // ===============================================
      // STEP 7: Save Results
      // ===============================================
      log("STEP 7: Saving feature vectors to results directory")
      
      val saveStartTime = System.nanoTime()
      
      // Convert vectors to strings for text output
      val featuresAsString = transformedDF.select(
        $"text",
        $"features".cast("string").as("feature_vector")
      )
      
      // Save main results as text file (requirement: results/lab17_pipeline_output.txt)
      val outputPath = "results/lab17_pipeline_output.txt"
      log(s"Saving feature vectors to: $outputPath")
      
      // Collect and save manually to avoid Hadoop issues
      val featureResults = featuresAsString.collect()
      val outputFile = new PrintWriter(new File(outputPath))
      
      try {
        featureResults.foreach { row =>
          outputFile.println(s"Text: ${row.getString(0).take(100)}...")
          outputFile.println(s"Features: ${row.getString(1)}")
          outputFile.println("-" * 80)
        }
      } finally {
        outputFile.close()
      }
      
      val saveDuration = (System.nanoTime() - saveStartTime) / 1e9d
      log(f"Feature vectors saved to $outputPath in $saveDuration%.2f seconds")
      
      // ===============================================
      // STEP 8: Generate Statistics and Summary
      // ===============================================
      log("STEP 8: Computing pipeline statistics")
      
      val vocabSize = transformedDF
        .select(explode($"filtered_tokens").as("word"))
        .filter(length($"word") > 1)
        .distinct()
        .count()
      
      val avgTokensPerDoc = transformedDF
        .select(size($"tokens").as("token_count"))
        .agg(avg("token_count"))
        .collect()(0)(0).asInstanceOf[Double]
      
      val avgFilteredTokensPerDoc = transformedDF
        .select(size($"filtered_tokens").as("filtered_count"))
        .agg(avg("filtered_count"))
        .collect()(0)(0).asInstanceOf[Double]
      
      // Get feature vector statistics
      val firstVector = transformedDF.select("features").first().getAs[org.apache.spark.ml.linalg.Vector](0)
      val vectorSparsity = (1.0 - firstVector.numNonzeros.toDouble / firstVector.size) * 100
      
      log("=== PIPELINE STATISTICS ===")
      log(s"Total documents processed: $finalRecordCount")
      log(s"Unique vocabulary size: $vocabSize")
      log(f"Average tokens per document: $avgTokensPerDoc%.2f")
      log(f"Average filtered tokens per document: $avgFilteredTokensPerDoc%.2f")
      log(s"Feature vector dimensions: $numFeatures")
      log(s"First vector non-zero elements: ${firstVector.numNonzeros}")
      log(f"Vector sparsity: $vectorSparsity%.2f%%")
      
      // Show sample of final results
      log("Sample of final processed data:")
      transformedDF.select($"text", $"tokens", $"filtered_tokens")
        .show(3, truncate = true)
      
      // ===============================================
      // STEP 9: Cleanup
      // ===============================================
      log("STEP 9: Cleaning up resources")
      spark.stop()
      
      val endTime = LocalDateTime.now()
      val totalDuration = java.time.Duration.between(startTime, endTime)
      val minutes = totalDuration.toMinutes
      val seconds = totalDuration.toSeconds % 60
      
      log(f"Pipeline completed successfully at: ${endTime.format(formatter)}")
      log(f"Total execution time: $minutes minutes $seconds seconds")
      log("=== PIPELINE EXECUTION SUMMARY ===")
      log(s"Data loaded: $recordCount records")
      log(s"Data processed: $finalRecordCount records")
      log(s"Feature vectors saved to: $outputPath")
      log(s"Log file saved to: ${logFile.getAbsolutePath}")
      
    } // End of Try block
    
    // Handle success/failure
    pipelineResult match {
      case Success(_) => 
        log("NLP DATA PIPELINE COMPLETED SUCCESSFULLY!")
        
      case Failure(exception) => 
        log(s"ERROR: Pipeline failed with exception: ${exception.getMessage}")
        log(s"Exception type: ${exception.getClass.getSimpleName}")
        log("Stack trace:")
        exception.getStackTrace.take(10).foreach(line => log(s"  $line"))
        
        // Re-throw the exception for proper error handling
        throw exception
    }
    
    // Always close the log writer
    try {
      logWriter.close()
      println(s"\nComplete log saved to: ${logFile.getAbsolutePath}")
      println(s"Results saved to: results/lab17_pipeline_output.txt")
      println(s"Check the log file for detailed pipeline statistics")
    } catch {
      case _: Exception => // Ignore errors when closing log writer
    }
  }
}