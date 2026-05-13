# spark/predict.py
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F


def predict_reviews(reviews_json_path: str, output_mongo_uri: str):
    spark = SparkSession.builder \
        .appName("FakeReviewPredictor") \
        .config("spark.mongodb.output.uri", output_mongo_uri) \
        .getOrCreate()

    model = PipelineModel.load(
        "hdfs://namenode:9000/user/models/fake_review_model"
    )
    df = spark.read.json(reviews_json_path)

    fill_cols = {
        "word_count": 0, "caps_ratio": 0.0, "exclamation_count": 0,
        "helpful_votes": 0, "rating": 3.0, "reviewer_fraud_score": 0.0,
        "rating_sentiment_mismatch": 0, "body": ""
    }
    df = df.fillna(fill_cols)

    predictions = model.transform(df)
    result = predictions.select(
        "review_id", "product_asin", "reviewer_id",
        "rating", "body", "label",
        "prediction",
        F.col("probability").getItem(1).alias("fake_probability")
    )

    # Write to MongoDB
    result.write \
        .format("mongo") \
        .mode("append") \
        .option("uri", output_mongo_uri) \
        .save()

    spark.stop()