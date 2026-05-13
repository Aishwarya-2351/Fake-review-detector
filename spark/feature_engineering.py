# spark/feature_engineering.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF,
    VectorAssembler, StandardScaler, StringIndexer
)
from pyspark.ml import Pipeline


def build_feature_pipeline():
    """
    Builds feature engineering pipeline.
    Features:
      - TF-IDF on review text
      - Numeric: word_count, caps_ratio, exclamation_count,
                 helpful_votes, rating, reviewer_fraud_score
      - Derived: rating_sentiment_mismatch, spam_flag
    """
    tokenizer = Tokenizer(inputCol="body", outputCol="words")
    stopwords_remover = StopWordsRemover(
        inputCol="words", outputCol="filtered_words"
    )
    hashing_tf = HashingTF(
        inputCol="filtered_words", outputCol="raw_tf", numFeatures=5000
    )
    idf = IDF(inputCol="raw_tf", outputCol="tfidf_features", minDocFreq=2)

    numeric_cols = [
        "word_count", "caps_ratio", "exclamation_count",
        "helpful_votes", "rating", "reviewer_fraud_score",
        "rating_sentiment_mismatch"
    ]
    assembler = VectorAssembler(
        inputCols=["tfidf_features"] + numeric_cols,
        outputCol="raw_features"
    )
    scaler = StandardScaler(
        inputCol="raw_features", outputCol="features",
        withStd=True, withMean=False
    )

    return Pipeline(stages=[
        tokenizer, stopwords_remover, hashing_tf,
        idf, assembler, scaler
    ])