-- hive/create_tables.hql

CREATE DATABASE IF NOT EXISTS reviews_db;
USE reviews_db;

DROP TABLE IF EXISTS raw_reviews;
CREATE EXTERNAL TABLE raw_reviews (
  review_id       STRING,
  product_asin    STRING,
  reviewer_id     STRING,
  reviewer_name   STRING,
  rating          DOUBLE,
  title           STRING,
  body            STRING,
  review_date     STRING,
  verified_purchase BOOLEAN,
  helpful_votes   INT,
  spam_flag       BOOLEAN,
  word_count      INT,
  caps_ratio      DOUBLE,
  exclamation_count INT,
  reviewer_fraud_score DOUBLE
)
ROW FORMAT SERDE 'org.apache.hive.hcatalog.data.JsonSerDe'
STORED AS TEXTFILE
LOCATION '/user/reviews/mapreduce_output/';

DROP TABLE IF EXISTS review_sentiments;
CREATE TABLE review_sentiments AS
SELECT
  review_id,
  product_asin,
  reviewer_id,
  rating,
  body,
  word_count,
  spam_flag,
  reviewer_fraud_score,
  -- Positive word count (simplified lexicon approach in HiveQL)
  (CASE WHEN lower(body) RLIKE '.*(great|excellent|amazing|perfect|love|best|awesome|fantastic).*'
        THEN 1 ELSE 0 END) AS positive_signal,
  (CASE WHEN lower(body) RLIKE '.*(terrible|awful|horrible|worst|hate|broken|useless|scam).*'
        THEN 1 ELSE 0 END) AS negative_signal,
  -- Rating-sentiment mismatch: high rating + negative words = suspicious
  (CASE WHEN rating >= 4.0
        AND lower(body) RLIKE '.*(terrible|awful|horrible|worst|hate|broken|useless).*'
        THEN 1 ELSE 0 END) AS rating_sentiment_mismatch,
  review_date
FROM raw_reviews
WHERE body IS NOT NULL AND LENGTH(body) > 10;