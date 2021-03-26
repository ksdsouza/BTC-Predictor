package ca.uwaterloo.CounterPredictor

import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

import java.text.SimpleDateFormat
import java.util.Calendar

object CounterPredictor {
  val sdf = new SimpleDateFormat("yyyy-MM-dd")

  def getNextDay(date: String): String = {
    val c = Calendar.getInstance
    c.setTime(sdf.parse(date))
    c.add(Calendar.DATE, 1)
    sdf.format(c.getTime)
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.getOrCreate()
    val sc = SparkContext.getOrCreate()
    import spark.implicits._

    val tweetsDF = (
      spark.read
        .option("header", "true")
        .option("delimiter", ";")
        .option("multiLine", "true")
        .option("quote", "\"")
        .option("escape", "\"")
        .csv("tweets.csv")
        .drop("id", "user", "fullname", "url", "replies", "likes", "retweets")
        .drop()
        .map(row => (row(0).toString.take(10), row(1).toString))
        .withColumnRenamed("_1", "t_timestamp")
        .withColumnRenamed("_2", "t_msg")
        .rdd
        .filter(row => !row(0).toString.startsWith("2019-"))
      )

    val prices = (
      spark.read
        .option("header", "true")
        .option("delimiter", ",")
        .option("multiLine", "true")
        .option("quote", "\"")
        .option("escape", "\"")
        .csv("BTC-USD.csv")
        .drop("Open", "High", "Close", "Adj Close", "Volume")
        .drop()
        .filter(row => !row(0).toString.equals("null") && !row(1).toString.equals("null"))
        .map(row => row(0).toString -> row(1).toString.toDouble)
        .collect()
        .toMap
      )
    val dateToPrice = sc.broadcast(prices)

    //    tweetsDF.createOrReplaceTempView("tweets")

    val r = (
      tweetsDF
        .filter(row => {
          val timestamp = row(0).toString
          dateToPrice.value.contains(timestamp)
        })
        .map(row => (row(0).toString, 1))
        .groupByKey()
        .map({ case (timestamp, messageCounts) => (timestamp, messageCounts.size) })
        .flatMap({ case (timestamp, numMessages) =>
          val nextDayTimestamp = getNextDay(timestamp)
          if (dateToPrice.value.contains(nextDayTimestamp)) {
            val price = dateToPrice.value(timestamp)
            val nextPrice = dateToPrice.value(nextDayTimestamp)
            List((nextPrice, Vectors.dense(price, numMessages)))
          }
          else List()
        })
      )

    val z = r.toDF("label", "features")
    println(z.take(3).mkString("Array(", ", ", ")"))
    println(z.show())

    val Array(train, test) = z.randomSplit(Array(0.8, 0.2))

    val linearRegression = new LinearRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(400)
      .setRegParam(0.25)
    //      .setElasticNetParam(1)
    val model = linearRegression.fit(train)
    val summary = model.evaluate(test)
    println(f"Root Mean Squared Error: ${summary.rootMeanSquaredError} MSE: ${summary.meanSquaredError}")

    model.write.overwrite().save("./model")
  }
}
