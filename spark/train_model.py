# spark/train_model.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier, RandomForestClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from feature_engineering import build_feature_pipeline
import json


def main():
    spark = SparkSession.builder \
        .appName("FakeReviewClassifier") \
        .master("spark://spark-master:7077") \
        .config("spark.mongodb.output.uri",
                "mongodb://mongo:27017/reviews_db.model_results") \
        .getOrCreate()

    # Load processed data from HDFS
    df = spark.read.json("hdfs://namenode:9000/user/reviews/mapreduce_output/")

    # Create label: 1 = fake, 0 = genuine
    # Ground truth: use reviewer_fraud_score > 0.5 as proxy label
    # (In production: use manually labelled dataset)
    df = df.withColumn(
        "label",
        F.when(F.col("reviewer_fraud_score") > 0.5, 1.0).otherwise(0.0)
    )

    # Fill nulls
    fill_cols = {
        "word_count": 0, "caps_ratio": 0.0, "exclamation_count": 0,
        "helpful_votes": 0, "rating": 3.0, "reviewer_fraud_score": 0.0,
        "rating_sentiment_mismatch": 0, "body": ""
    }
    df = df.fillna(fill_cols)

    # Handle class imbalance with oversampling
    fake_count = df.filter(F.col("label") == 1.0).count()
    genuine_count = df.filter(F.col("label") == 0.0).count()
    ratio = genuine_count / max(fake_count, 1)
    fake_oversampled = df.filter(F.col("label") == 1.0).sample(
        withReplacement=True, fraction=min(ratio, 5.0)
    )
    df_balanced = df.filter(F.col("label") == 0.0).union(fake_oversampled)

    train_df, test_df = df_balanced.randomSplit([0.8, 0.2], seed=42)

    # Build pipeline
    feature_pipeline = build_feature_pipeline()
    gbt = GBTClassifier(
        featuresCol="features", labelCol="label",
        maxIter=50, maxDepth=5, stepSize=0.1
    )
    full_pipeline = Pipeline(
        stages=feature_pipeline.getStages() + [gbt]
    )

    # Hyperparameter tuning
    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter, [30, 50])
        .addGrid(gbt.maxDepth, [4, 6])
        .build()
    )
    evaluator = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC"
    )
    cv = CrossValidator(
        estimator=full_pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=2
    )

    print("Training model...")
    cv_model = cv.fit(train_df)
    best_model = cv_model.bestModel

    # Evaluate
    predictions = best_model.transform(test_df)
    auc = evaluator.evaluate(predictions)

    mc_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction"
    )
    accuracy = mc_eval.evaluate(predictions, {mc_eval.metricName: "accuracy"})
    f1 = mc_eval.evaluate(predictions, {mc_eval.metricName: "f1"})
    precision = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedPrecision"})
    recall = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedRecall"})

    metrics = {
        "auc_roc": round(auc, 4),
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }
    print("Model metrics:", json.dumps(metrics, indent=2))

    # Save model to HDFS
    best_model.write().overwrite().save(
        "hdfs://namenode:9000/user/models/fake_review_model"
    )
    print("Model saved to HDFS.")

    spark.stop()


if __name__ == "__main__":
    main()