# run_hive_local.py
# Simulates Hive queries using pandas (same logic as HiveQL)
import json
import pandas as pd
import re
import os

print("="*55)
print("  HIVE SENTIMENT ANALYSIS — LOCAL SIMULATION")
print("="*55)

# Load processed reviews
reviews = []
with open("hadoop/map/processed_reviews.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            reviews.append(json.loads(line))

df = pd.DataFrame(reviews)
print(f"\nLoaded {len(df)} reviews into Hive simulation")

# ── HiveQL: CREATE TABLE review_sentiments ────────────
POSITIVE = re.compile(r"\b(great|excellent|amazing|perfect|love|best|awesome|fantastic)\b", re.I)
NEGATIVE = re.compile(r"\b(terrible|awful|horrible|worst|hate|broken|useless|scam)\b", re.I)

df["positive_signal"]      = df["body"].apply(lambda x: 1 if POSITIVE.search(str(x)) else 0)
df["negative_signal"]      = df["body"].apply(lambda x: 1 if NEGATIVE.search(str(x)) else 0)
df["rating_sentiment_mismatch"] = df.apply(
    lambda r: 1 if float(r.get("rating", 3)) >= 4.0
              and NEGATIVE.search(str(r.get("body", ""))) else 0, axis=1
)

print("\n[HIVE TABLE] review_sentiments — created ✅")

# ── HiveQL: suspicious_reviewers ──────────────────────
reviewer_group = df.groupby("reviewer_id").agg(
    review_count        = ("reviewer_id", "count"),
    avg_rating          = ("rating", "mean"),
    spam_count          = ("spam_flag", "sum"),
    avg_fraud_score     = ("reviewer_fraud_score", "mean"),
    mismatch_count      = ("rating_sentiment_mismatch", "sum"),
).reset_index()

suspicious_reviewers = reviewer_group[
    reviewer_group["review_count"] >= 2
].sort_values("avg_fraud_score", ascending=False)

print("\n[HIVE QUERY] suspicious_reviewers — Top 10:")
print("-"*55)
print(suspicious_reviewers.head(10).to_string(index=False))

# ── HiveQL: product_fraud_summary ─────────────────────
product_group = df.groupby("product_asin").agg(
    total_reviews    = ("product_asin", "count"),
    avg_rating       = ("rating", "mean"),
    spam_reviews     = ("spam_flag", "sum"),
    avg_fraud_score  = ("reviewer_fraud_score", "mean"),
    sentiment_mismatches = ("rating_sentiment_mismatch", "sum"),
).reset_index()

product_group["spam_percentage"] = (
    product_group["spam_reviews"] / product_group["total_reviews"] * 100
).round(2)

print("\n[HIVE QUERY] product_fraud_summary:")
print("-"*55)
print(product_group.sort_values("spam_percentage", ascending=False).to_string(index=False))

# ── HiveQL: daily_review_bursts ───────────────────────
df["date_clean"] = df["date"].apply(
    lambda x: str(x).split(" on ")[-1].strip() if " on " in str(x) else "Unknown"
)
daily_group = df.groupby(["product_asin", "date_clean"]).agg(
    daily_count      = ("product_asin", "count"),
    daily_avg_rating = ("rating", "mean"),
    daily_spam_count = ("spam_flag", "sum"),
).reset_index()

bursts = daily_group[daily_group["daily_count"] >= 3].sort_values("daily_count", ascending=False)
print("\n[HIVE QUERY] daily_review_bursts (count >= 3):")
print("-"*55)
print(bursts.head(10).to_string(index=False))

# ── Save results ──────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
suspicious_reviewers.to_csv("outputs/hive_suspicious_reviewers.csv", index=False)
product_group.to_csv("outputs/hive_product_fraud_summary.csv", index=False)
bursts.to_csv("outputs/hive_daily_bursts.csv", index=False)

print(f"""
{'='*55}
  ✅ HIVE SIMULATION COMPLETE
{'='*55}
  Outputs saved to outputs/ folder:
  - hive_suspicious_reviewers.csv
  - hive_product_fraud_summary.csv
  - hive_daily_bursts.csv
{'='*55}
""")