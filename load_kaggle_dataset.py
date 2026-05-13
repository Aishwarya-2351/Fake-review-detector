# load_kaggle_dataset.py
import pandas as pd
import json
import os
import glob
import uuid
import re
from datetime import datetime

print("="*55)
print("  LOADING KAGGLE AMAZON REVIEWS DATASET")
print("="*55)

# ── Find the CSV file anywhere in project ─────────────
print("\nSearching for CSV files...")
csv_files = (
    glob.glob("dataset/*.csv")
    + glob.glob("*.csv")
    + glob.glob("**/*.csv", recursive=True)
)

# Remove venv and node_modules paths
csv_files = [
    f for f in csv_files
    if "venv" not in f
    and "node_modules" not in f
    and "site-packages" not in f
]

if not csv_files:
    print("\nERROR: No CSV files found anywhere in project folder.")
    print("Make sure you extracted the ZIP into your project folder.")
    exit()

print(f"\nFound {len(csv_files)} CSV file(s):")
for f in csv_files:
    size = os.path.getsize(f) / (1024 * 1024)
    print(f"  {f}  ({size:.2f} MB)")

# ── Load all CSV files ────────────────────────────────
dfs = []
for csv_file in csv_files:
    print(f"\nLoading: {csv_file}")
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        print(f"  Rows    : {df.shape[0]}")
        print(f"  Columns : {df.shape[1]}")
        print(f"  Col list: {list(df.columns)}")
        dfs.append(df)
    except Exception as e:
        print(f"  SKIPPED — error reading file: {e}")

if not dfs:
    print("\nERROR: Could not load any CSV files.")
    exit()

df = pd.concat(dfs, ignore_index=True)
print(f"\nCombined shape: {df.shape}")

# ── Detect column names automatically ─────────────────
# Handle both possible column naming formats
cols = list(df.columns)
print(f"\nAll columns: {cols}")

# Map possible column name variations
def find_col(df, options):
    for opt in options:
        if opt in df.columns:
            return opt
    return None

text_col     = find_col(df, ["reviews.text", "review_text", "reviewText", "text", "body"])
rating_col   = find_col(df, ["reviews.rating", "rating", "overall", "stars"])
title_col    = find_col(df, ["reviews.title", "review_title", "summary", "title"])
username_col = find_col(df, ["reviews.username", "reviewerName", "username", "reviewer"])
date_col     = find_col(df, ["reviews.date", "reviewTime", "date", "review_date"])
helpful_col  = find_col(df, ["reviews.numHelpful", "helpful", "helpful_votes"])
asin_col     = find_col(df, ["asins", "asin", "product_id", "id"])
name_col     = find_col(df, ["name", "product_name", "productTitle"])
brand_col    = find_col(df, ["brand", "manufacturer"])
cat_col      = find_col(df, ["primaryCategories", "category", "categories"])

print(f"\nDetected columns:")
print(f"  Review text  : {text_col}")
print(f"  Rating       : {rating_col}")
print(f"  Title        : {title_col}")
print(f"  Username     : {username_col}")
print(f"  Date         : {date_col}")
print(f"  Helpful      : {helpful_col}")
print(f"  ASIN         : {asin_col}")

if not text_col or not rating_col:
    print("\nERROR: Could not find review text or rating column.")
    print("Your columns are:", cols)
    exit()

# ── Clean data ────────────────────────────────────────
print("\nCleaning data...")
df = df.dropna(subset=[text_col, rating_col])
df = df[df[text_col].astype(str).str.strip() != ""]
df = df[df[text_col].astype(str).str.strip() != "nan"]
print(f"After cleaning: {len(df)} reviews")

# Sample 5000 for speed
if len(df) > 5000:
    df = df.sample(n=5000, random_state=42)
    print(f"Sampled to 5000 reviews")

df = df.reset_index(drop=True)

# ── Fake label derivation ─────────────────────────────
print("\nDeriving fake/genuine labels...")

SPAM_REGEX = re.compile(
    r"\b(buy now|click here|visit our|free|discount|promo|coupon|check out)\b",
    re.IGNORECASE
)
NEGATIVE_REGEX = re.compile(
    r"\b(terrible|awful|horrible|worst|hate|broken|useless|scam|garbage|junk|fake|fraud)\b",
    re.IGNORECASE
)
POSITIVE_REGEX = re.compile(
    r"\b(great|excellent|amazing|perfect|love|best|awesome|fantastic|outstanding)\b",
    re.IGNORECASE
)

def derive_fake_label(body, rating):
    body   = str(body)
    try:
        rating = float(rating)
    except (ValueError, TypeError):
        rating = 3.0

    score = 0.0
    words = body.split()

    # Very short + extreme rating
    if len(words) < 5 and rating in [1.0, 5.0]:
        score += 0.35

    # Too many exclamations
    if body.count("!") > 4:
        score += 0.20

    # High caps ratio
    caps_ratio = sum(1 for c in body if c.isupper()) / max(len(body), 1)
    if caps_ratio > 0.30:
        score += 0.20

    # Spam words
    if SPAM_REGEX.search(body):
        score += 0.30

    # Sentiment mismatch
    if rating >= 4.0 and NEGATIVE_REGEX.search(body):
        score += 0.35

    # Very generic 5-star
    if len(words) < 8 and rating == 5.0:
        score += 0.15

    # Repetitive positive words only, no specifics
    pos_count = len(POSITIVE_REGEX.findall(body))
    if pos_count >= 3 and len(words) < 15:
        score += 0.20

    return 1 if score >= 0.5 else 0

df["is_fake_ground_truth"] = df.apply(
    lambda row: derive_fake_label(
        row[text_col],
        row[rating_col]
    ),
    axis=1
)

# ── Convert to project format ─────────────────────────
print("Converting to project format...")
reviews = []

for _, row in df.iterrows():
    # Rating
    try:
        rating = float(row[rating_col])
        rating = max(1.0, min(5.0, rating))
    except (ValueError, TypeError):
        rating = 3.0

    # Date
    try:
        dt = pd.to_datetime(str(row[date_col])) if date_col else None
        date_str = (
            f"Reviewed in the United States on "
            f"{dt.strftime('%B %d, %Y')}"
            if dt else
            "Reviewed in the United States on January 01, 2024"
        )
    except Exception:
        date_str = "Reviewed in the United States on January 01, 2024"

    # Helpful votes
    try:
        helpful = int(float(row[helpful_col])) if helpful_col else 0
    except (ValueError, TypeError):
        helpful = 0

    body = str(row[text_col]).strip()

    review = {
        "review_id"           : str(uuid.uuid4())[:16].upper(),
        "product_asin"        : str(row[asin_col])[:20] if asin_col else "UNKNOWN",
        "reviewer_id"         : str(row[username_col])[:20] if username_col else "UNKNOWN",
        "reviewer_name"       : str(row[username_col]) if username_col else "Anonymous",
        "rating"              : rating,
        "title"               : str(row[title_col])[:100] if title_col else "",
        "body"                : body,
        "date"                : date_str,
        "verified_purchase"   : helpful > 0,
        "helpful_votes"       : helpful,
        "product_name"        : str(row[name_col])[:100] if name_col else "",
        "brand"               : str(row[brand_col])[:50] if brand_col else "",
        "category"            : str(row[cat_col])[:50] if cat_col else "",
        "scraped_at"          : datetime.utcnow().isoformat(),
        "is_fake_ground_truth": int(row["is_fake_ground_truth"]),
    }
    reviews.append(review)

# ── Save ──────────────────────────────────────────────
os.makedirs("scraper", exist_ok=True)
output_path = "scraper/raw_reviews.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(reviews, f, indent=2)

# ── Print summary ─────────────────────────────────────
genuine    = sum(1 for r in reviews if r["is_fake_ground_truth"] == 0)
fake       = sum(1 for r in reviews if r["is_fake_ground_truth"] == 1)
ratings    = [r["rating"] for r in reviews]
avg_rating = sum(ratings) / len(ratings)

# Rating distribution
rating_dist = {}
for r in reviews:
    star = int(r["rating"])
    rating_dist[star] = rating_dist.get(star, 0) + 1

print(f"""
{'='*55}
  KAGGLE DATASET LOADED SUCCESSFULLY
{'='*55}
  Total Reviews   : {len(reviews)}
  Genuine Reviews : {genuine}
  Fake Reviews    : {fake}
  Fake Percentage : {round(fake / max(len(reviews), 1) * 100, 2)}%
  Avg Rating      : {round(avg_rating, 2)} ⭐
{'='*55}
  Rating Distribution:
    ⭐⭐⭐⭐⭐ 5 stars : {rating_dist.get(5, 0)}
    ⭐⭐⭐⭐  4 stars : {rating_dist.get(4, 0)}
    ⭐⭐⭐   3 stars : {rating_dist.get(3, 0)}
    ⭐⭐     2 stars : {rating_dist.get(2, 0)}
    ⭐      1 star  : {rating_dist.get(1, 0)}
{'='*55}
  Saved to: {output_path}
{'='*55}

  Next steps:
  1. python run_mapreduce.py
  2. python train_model_local.py
""")