# hadoop/mapreduce/spam_filter_reducer.py
#!/usr/bin/env python3
"""
Reducer: groups reviews by reviewer_id.
Flags reviewers with burst activity (>5 reviews in a day)
or extreme rating patterns (all 5s or all 1s).
"""
import sys
import json
from collections import defaultdict
from datetime import datetime

current_reviewer = None
reviews_buffer = []

def process_reviewer(reviewer_id, reviews):
    """Apply reviewer-level fraud signals."""
    if not reviews:
        return

    ratings = [r.get("rating", 3) for r in reviews]
    avg_rating = sum(ratings) / len(ratings)
    rating_std = (
        sum((r - avg_rating) ** 2 for r in ratings) / len(ratings)
    ) ** 0.5

    # Burst detection: parse dates and check 24h windows
    dates = []
    for r in reviews:
        raw_date = r.get("date", "")
        try:
            # Amazon format: "Reviewed in the United States on January 1, 2024"
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

    for review in reviews:
        review["reviewer_flags"] = reviewer_flags
        review["reviewer_fraud_score"] = (
            (0.3 if reviewer_flags["all_extreme_ratings"] else 0)
            + (0.3 if reviewer_flags["burst_activity"] else 0)
            + (0.2 if reviewer_flags["unverified_ratio"] > 0.8 else 0)
            + (0.2 if review.get("spam_flag", False) else 0)
        )
        print(json.dumps(review))


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    parts = line.split("\t", 1)
    if len(parts) != 2:
        continue

    reviewer_id, review_json = parts
    try:
        review = json.loads(review_json)
    except json.JSONDecodeError:
        continue

    if reviewer_id != current_reviewer:
        if current_reviewer is not None:
            process_reviewer(current_reviewer, reviews_buffer)
        current_reviewer = reviewer_id
        reviews_buffer = []

    reviews_buffer.append(review)

if current_reviewer is not None:
    process_reviewer(current_reviewer, reviews_buffer)