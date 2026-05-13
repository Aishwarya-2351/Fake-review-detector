# hadoop/mapreduce/spam_filter_mapper.py
#!/usr/bin/env python3
"""
Mapper: reads raw JSON reviews from HDFS, emits (reviewer_id, review_data)
for spam filtering. Filters out reviews with suspiciously short bodies,
extreme ratings with no helpful votes, and duplicate content signals.
"""
import sys
import json
import re

SPAM_PATTERNS = [
    r"\b(buy now|click here|visit our|check out our website)\b",
    r"\b(free|discount|promo code|coupon)\b",
    r"https?://",
]
SPAM_REGEX = re.compile("|".join(SPAM_PATTERNS), re.IGNORECASE)

MIN_BODY_LENGTH = 20

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        review = json.loads(line)
        body = review.get("body", "")
        rating = review.get("rating", 3)
        reviewer_id = review.get("reviewer_id", "UNKNOWN")

        # Basic spam signals
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

        # Key = reviewer_id so reducer can group by reviewer
        print(f"{reviewer_id}\t{json.dumps(review)}")

    except (json.JSONDecodeError, KeyError):
        continue