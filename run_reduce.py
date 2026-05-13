# run_mapreduce.py
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hadoop", "map"))

# ── Import mapper and reducer logic ──────────────────
from collections import defaultdict
import re
from datetime import datetime

SPAM_PATTERNS = [
    r"\b(buy now|click here|visit our|check out our website)\b",
    r"\b(free|discount|promo code|coupon)\b",
    r"https?://",
]
SPAM_REGEX = re.compile("|".join(SPAM_PATTERNS), re.IGNORECASE)
MIN_BODY_LENGTH = 20

def mapper(review):
    body = review.get("body", "")
    rating = review.get("rating", 3)
    reviewer_id = review.get("reviewer_id", "UNKNOWN")

    is_spam = (
        len(body) < MIN_BODY_LENGTH
        or bool(SPAM_REGEX.search(body))
        or (rating in [1.0, 5.0] and review.get("helpful_votes", 0) == 0
            and len(body) < 50)
    )

    review["spam_flag"] = is_spam
    review["word_count"] = len(body.split())
    review["has_url"] = bool(re.search(r"https?://", body))
    review["exclamation_count"] = body.count("!")
    review["caps_ratio"] = (
        sum(1 for c in body if c.isupper()) / max(len(body), 1)
    )
    return reviewer_id, review

def reducer(reviewer_id, reviews):
    ratings = [r.get("rating", 3) for r in reviews]
    avg_rating = sum(ratings) / len(ratings)
    rating_std = (
        sum((r - avg_rating) ** 2 for r in ratings) / len(ratings)
    ) ** 0.5

    dates = []
    for r in reviews:
        raw_date = r.get("date", "")
        try:
            date_part = raw_date.split(" on ")[-1]
            dates.append(datetime.strptime(date_part, "%B %d, %Y"))
        except ValueError:
            pass

    dates.sort()
    burst_flag = False
    if len(dates) >= 5:
        for i in range(len(dates) - 4):
            delta = (dates[i + 4] - dates[i]).total_seconds() / 3600
            if delta <= 24:
                burst_flag = True
                break

    reviewer_flags = {
        "total_reviews": len(reviews),
        "avg_rating": round(avg_rating, 2),
        "rating_std": round(rating_std, 2),
        "all_extreme_ratings": all(r in [1.0, 5.0] for r in ratings),
        "burst_activity": burst_flag,
        "unverified_ratio": sum(
            1 for r in reviews if not r.get("verified_purchase", False)
        ) / len(reviews),
    }

    processed = []
    for review in reviews:
        review["reviewer_flags"] = reviewer_flags
        review["reviewer_fraud_score"] = round(min(
            (0.3 if reviewer_flags["all_extreme_ratings"] else 0)
            + (0.3 if reviewer_flags["burst_activity"] else 0)
            + (0.2 if reviewer_flags["unverified_ratio"] > 0.8 else 0)
            + (0.2 if review.get("spam_flag", False) else 0),
            1.0
        ), 3)
        processed.append(review)
    return processed

# ── Main pipeline ─────────────────────────────────────
print("Loading raw reviews...")
with open("scraper/raw_reviews.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Total reviews loaded: {len(raw_data)}")

# MAP phase
print("Running mapper...")
mapped = defaultdict(list)
for review in raw_data:
    reviewer_id, processed_review = mapper(review)
    mapped[reviewer_id].append(processed_review)

print(f"Unique reviewers found: {len(mapped)}")

# REDUCE phase
print("Running reducer...")
all_processed = []
for reviewer_id, reviews in mapped.items():
    reduced = reducer(reviewer_id, reviews)
    all_processed.extend(reduced)

# Save output
output_path = "hadoop/map/processed_reviews.json"
with open(output_path, "w", encoding="utf-8") as f:
    for item in all_processed:
        f.write(json.dumps(item) + "\n")

# ── Print summary ─────────────────────────────────────
fake = [r for r in all_processed if r.get("reviewer_fraud_score", 0) >= 0.5]
genuine = [r for r in all_processed if r.get("reviewer_fraud_score", 0) < 0.5]
spam = [r for r in all_processed if r.get("spam_flag", False)]

print("\n" + "="*50)
print("   MAPREDUCE COMPLETED SUCCESSFULLY")
print("="*50)
print(f"  Total processed : {len(all_processed)}")
print(f"  Fake reviews    : {len(fake)}")
print(f"  Genuine reviews : {len(genuine)}")
print(f"  Spam flagged    : {len(spam)}")
print("="*50)
print(f"\n Output saved to: {output_path}")