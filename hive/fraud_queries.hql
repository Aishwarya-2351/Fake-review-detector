-- hive/fraud_queries.hql
USE reviews_db;

-- Top suspicious reviewers
CREATE TABLE IF NOT EXISTS suspicious_reviewers AS
SELECT
  reviewer_id,
  COUNT(*) AS review_count,
  AVG(rating) AS avg_rating,
  SUM(spam_flag) AS spam_count,
  AVG(reviewer_fraud_score) AS avg_fraud_score,
  SUM(rating_sentiment_mismatch) AS mismatches
FROM review_sentiments
GROUP BY reviewer_id
HAVING COUNT(*) > 3
ORDER BY avg_fraud_score DESC;

-- Product-level fraud summary
CREATE TABLE IF NOT EXISTS product_fraud_summary AS
SELECT
  product_asin,
  COUNT(*) AS total_reviews,
  AVG(rating) AS avg_rating,
  STDDEV(rating) AS rating_stddev,
  SUM(spam_flag) AS spam_reviews,
  ROUND(SUM(spam_flag) / COUNT(*) * 100, 2) AS spam_percentage,
  SUM(rating_sentiment_mismatch) AS sentiment_mismatches,
  AVG(reviewer_fraud_score) AS avg_fraud_exposure
FROM review_sentiments
GROUP BY product_asin
ORDER BY spam_percentage DESC;

-- Daily review burst detection per product
CREATE TABLE IF NOT EXISTS daily_review_bursts AS
SELECT
  product_asin,
  review_date,
  COUNT(*) AS daily_count,
  AVG(rating) AS daily_avg_rating,
  SUM(spam_flag) AS daily_spam_count
FROM review_sentiments
GROUP BY product_asin, review_date
HAVING COUNT(*) > 10
ORDER BY daily_count DESC;