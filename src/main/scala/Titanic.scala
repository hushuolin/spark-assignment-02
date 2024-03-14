import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Titanic extends App {
  // Initialize SparkSession
  val spark = SparkSession.builder()
    .appName("Titanic Data Analysis")
    .config("spark.master", "local")
    .getOrCreate()

  import spark.implicits._

  // Load the Titanic dataset
  val df = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("train.csv")

  // 1) Average Ticket Fare for Each Ticket Class
  val averageFareByClass = df.groupBy("Pclass")
    .agg(avg("Fare").alias("Average_Fare"))
    .orderBy("Pclass")

  // 2) Survival Percentage for Each Ticket Class
  val survivalRateByClass = df.groupBy("Pclass")
    .agg((sum("Survived") / count("Survived") * 100).alias("Survival_Rate"))
    .orderBy(desc("Survival_Rate"))

  // 3) Find Passengers Possibly be Rose
  val possibleRoses = df.filter(
      col("Pclass") === 1 &&
      col("Sex") === "female" &&
      col("Age") === 17.0 &&
        col("Parch") === 1
  )

  // Counting the number of possible Roses
  val numberOfPossibleRoses = possibleRoses.count()

  // 4) Find Passengers Possibly be Jack
  val possibleJacks = df.filter(
      col("Pclass") === 3 &&
      col("Sex") === "male" &&
      col("Age").isin(17.0, 18.0) &&
      col("Parch") === 0 &&
        col("SibSp") === 0
  )

  // Counting the number of possible Roses
  val numberOfPossibleJacks = possibleJacks.count()

  // 5) Relation between Ages and Ticket Fare, Most Possibly Survived Group
  val ageGroup = when($"Age" <= 10, "1-10")
    .when($"Age" > 10 && $"Age" <= 20, "11-20")
    .when($"Age" > 20 && $"Age" <= 30, "21-30")
    .when($"Age" > 30 && $"Age" <= 40, "31-40")
    .when($"Age" > 40 && $"Age" <= 50, "41-50")
    .when($"Age" > 50 && $"Age" <= 60, "51-60")
    .when($"Age" > 60, "61+")
    .otherwise("Unknown")

  val dfWithAgeGroup = df.withColumn("Age_Group", ageGroup)

  val fareByAgeGroup = dfWithAgeGroup.groupBy("Age_Group")
    .agg(avg("Fare").alias("Average_Fare"))
    .orderBy("Age_Group")

  val survivalByAgeGroup = dfWithAgeGroup.groupBy("Age_Group")
    .agg((sum("Survived") / count("Survived") * 100).alias("Survival_Rate"))
    .orderBy("Age_Group")

  // Show Results
  // 1) The average fare of 1st class is about 84.15, the 2nd class is about 20.66, the 3rd class is about 13.68
  println("Average Ticket Fare for Each Ticket Class:")
  averageFareByClass.show()

  // 2) The survival percentage of 1st class is about 62.96%, the 2nd class is about 47.28%, the 3rd class is about 24.24%
  // The 1st class has the highest survival rate
  println("Survival Percentage for Each Ticket Class:")
  survivalRateByClass.show()

  // 3) The number of possible Roses is 0
  println(s"Number of possible Roses: $numberOfPossibleRoses")

  // 4) The number of possible Jacks is 9
  println(s"Number of possible Jacks: $numberOfPossibleJacks")

  // 5) For Ages between 1-30, younger passengers have higher fare in average
  // For Ages between 31-60, older passengers have higher fare in average
  println("Relation Between Ages and Ticket Fare:")
  fareByAgeGroup.show()

  // The age group '1-10' most likely survived, whose survival rate is 59.375%
  println("Survival Rate by Age Group:")
  survivalByAgeGroup.show()
}



