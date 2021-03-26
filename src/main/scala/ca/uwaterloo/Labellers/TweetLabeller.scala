package ca.uwaterloo.Labellers

import edu.stanford.nlp.ling.CoreAnnotations.{LemmaAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession

import java.util.Properties
import scala.collection.JavaConversions._

object TweetLabeller {

  def plainTextToLemmas(text: String, stopWords: Set[String]): Seq[String] = {
    val props = new Properties()
    props.put("annotators", "tokenize, ssplit, pos, lemma")
    val pipeline = new StanfordCoreNLP(props)
    val doc = new Annotation(text)
    pipeline.annotate(doc)
    val sentences = doc.get(classOf[SentencesAnnotation])

    for {
      sentence <- sentences
      token <- sentence.get(classOf[TokensAnnotation])
      lemma = token.get(classOf[LemmaAnnotation]).toLowerCase
      if lemma.length > 2 && !stopWords.contains(lemma)
    } yield lemma
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.getOrCreate()
    implicit val sc = SparkContext.getOrCreate()
    import spark.implicits._
    val stop: Broadcast[Set[String]] = sc.broadcast(stopWords)

    val bearish = sc.textFile("data/bearish.tweets.txt")
      .map(tweet => plainTextToLemmas(tweet, stop.value))
      .filter(_.nonEmpty)
      .map(("bearish", _))
      .toDF("label", "words")


    val bullish = sc.textFile("data/bullish.tweets.txt")
      .map(tweet => plainTextToLemmas(tweet, stop.value))
      .filter(_.nonEmpty)
      .map(("bullish", _))
      .toDF("label", "words")

    val neutral = sc.textFile("data/neutral.tweets.txt")
      .map(tweet => plainTextToLemmas(tweet, stop.value))
      .filter(_.nonEmpty)
      .map(("neutral", _))
      .toDF("label", "words")

    val tweets = bearish
      .union(bullish)
      .union(neutral)
      .cache()

    val featurizedData = new HashingTF()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .transform(tweets)
      .cache()

    val labelledData = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
      .fit(featurizedData)
      .transform(featurizedData)
      .select("label", "features")
      .rdd
      .map(row => (row(0).toString, row(1).asInstanceOf[SparseVector].values))
      .map({ case (label, features: Array[Double]) =>
        val paddedFeatures = new DenseVector(
          features
            .map(_.toString.toDouble)
            .padTo(180, 0)
        )
        (label, paddedFeatures)
      })
      .cache()

    val Array(training, test) = labelledData.randomSplit(Array(0.8, 0.2), seed = 5L)


    val bearishTraining = training
      .map({ case (label, features) =>
        LabeledPoint(if (label == "bearish") 1 else 0, features)
      })
      .cache()

    val bullishTraining = training
      .map({ case (label, features) =>
        LabeledPoint(if (label == "bullish") 1 else 0, features)
      })
      .cache()

    val numIterations = 80
    val bearishModel = SVMWithSGD.train(bearishTraining, numIterations)
    bearishModel.clearThreshold()

    val bullishModel = SVMWithSGD.train(bullishTraining, numIterations)
    bullishModel.clearThreshold()

    //    val neutralModel = SVMWithSGD.train(neutralTraining, numIterations)
    //    neutralModel.clearThreshold()

    val scoreAndLabels = test.map { case (label, features) =>
      val bearishScore = bearishModel.predict(features)
      val bullishScore = bullishModel.predict(features)
      val maxScore = Math.max(bearishScore, bullishScore)

      if (bearishScore == maxScore) {
        (bearishScore, if (label == "bearish") 1d else 0d)
      }
      else if (bullishScore == maxScore) {
        (bullishScore, if (label == "bullish") 1d else 0d)
      }
      else {
        (1 - (bearishScore + bullishScore) / 2, if (label == "neutral") 1d else 0d)
      }
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    //    bearishModel.save(sc, "./bearishModel")
    //    bullishModel.save(sc, "./bullishModel")
    //    neutralModel.save(sc, "./neutralModel")

    println(s"Area under ROC = $auROC")
  }
}
