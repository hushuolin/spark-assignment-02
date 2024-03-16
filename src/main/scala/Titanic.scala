import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline

object Titanic extends App {

  // Initialize Spark session
  val spark = SparkSession.builder()
    .appName("Titanic Survival Prediction")
    .master("local[*]")
    .getOrCreate()

  // Load the dataset
  val trainDF = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("train.csv")

  // Fill missing values
  val ageMedian = trainDF.stat.approxQuantile("Age", Array(0.5), 0.001).head
  val fareMedian = trainDF.stat.approxQuantile("Fare", Array(0.5), 0.001).head
  val embarkedMode = trainDF.groupBy("Embarked").count().orderBy(desc("count")).first().getString(0)

  val filledDF = trainDF.na.fill(Map(
    "Age" -> ageMedian,
    "Fare" -> fareMedian,
    "Embarked" -> embarkedMode
  ))

  // Feature engineering
  val featureEngDF = filledDF
    .withColumn("FamilySize", col("SibSp") + col("Parch"))
    .withColumn("IsAlone", when(col("FamilySize") === 0, 1).otherwise(0))
    .withColumn("Sex", when(col("Sex") === "male", 0).otherwise(1))
    .withColumn("Embarked", when(col("Embarked") === "Q", 0).when(col("Embarked") === "S", 1).otherwise(2))
    .drop("PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch")

  // Split the data
  val Array(trainingData, testData) = featureEngDF.randomSplit(Array(0.8, 0.2))

  // Configure an ML pipeline
  val assembler = new VectorAssembler()
    .setInputCols(Array("Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone"))
    .setOutputCol("features")

  val logisticRegression = new LogisticRegression()
    .setLabelCol("Survived")
    .setFeaturesCol("features")

  val pipeline = new Pipeline().setStages(Array(assembler, logisticRegression))

  // Train the model
  val model = pipeline.fit(trainingData)

  // Make predictions
  val predictions = model.transform(testData)

  // Select example rows to display
  predictions.select("prediction", "Survived", "features").show(5)

  // Evaluate the model
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("Survived")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

  val accuracy = evaluator.evaluate(predictions)
  println(s"Accuracy: $accuracy")

  // Load the new passenger data
  val passengersDF = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("test.csv")

  // Feature engineering on the new passenger data
  val featurePassengerDF = passengersDF
    .na.fill(Map(
      "Age" -> ageMedian,
      "Fare" -> fareMedian,
      "Embarked" -> embarkedMode
    ))
    .withColumn("FamilySize", col("SibSp") + col("Parch"))
    .withColumn("IsAlone", when(col("FamilySize") === 0, 1).otherwise(0))
    .withColumn("Sex", when(col("Sex") === "male", 0).otherwise(1))
    .withColumn("Embarked", when(col("Embarked") === "Q", 0).when(col("Embarked") === "S", 1).otherwise(2))

  // Predict survival on the new data
  val predictionsForPassengers = model.transform(featurePassengerDF)

  // Show predictions along with PassengerId
  predictionsForPassengers.select("PassengerId", "name", "prediction").show()

  // Stop the Spark session
  spark.stop()
}
