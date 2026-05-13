# test_output.py
import json
from collections import defaultdict
from datetime import datetime

# ── Load processed reviews ────────────────────────────
print("\nLoading processed reviews...")
reviews = []
with open("hadoop/map/processed_reviews.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            reviews.append(json.loads(line))

print(f"Total reviews loaded: {len(reviews)}")

# ── Overall stats ─────────────────────────────────────
fake    = [r for r in reviews if r.get("reviewer_fraud_score", 0) >= 0.5]
genuine = [r for r in reviews if r.get("reviewer_fraud_score", 0) < 0.5]
spam    = [r for r in reviews if r.get("spam_flag", False)]

print(f"""
{'='*55}
   FAKE REVIEW DETECTOR — FULL OUTPUT REPORT
{'='*55}
  Total Reviews   : {len(reviews)}
  Fake Reviews    : {len(fake)}
  Genuine Reviews : {len(genuine)}
  Spam Flagged    : {len(spam)}
  Fake Percentage : {round(len(fake)/max(len(reviews),1)*100, 2)}%
{'='*55}
""")

# ── Top 5 suspicious reviews ──────────────────────────
print("TOP 5 MOST SUSPICIOUS REVIEWS:")
print("-" * 55)
sorted_reviews = sorted(
    reviews,
    key=lambda x: x.get("reviewer_fraud_score", 0),
    reverse=True
)
for i, r in enumerate(sorted_reviews[:5], 1):
    flags = r.get("reviewer_flags", {})
    print(f"""
[{i}] Reviewer ID   : {r.get('reviewer_id', 'N/A')}
    Product ASIN  : {r.get('product_asin', 'N/A')}
    Rating        : {'⭐' * int(r.get('rating', 0))}  ({r.get('rating')})
    Fraud Score   : {r.get('reviewer_fraud_score', 0):.2f}
    Spam Flag     : {r.get('spam_flag', False)}
    Word Count    : {r.get('word_count', 0)}
    Caps Ratio    : {r.get('caps_ratio', 0):.2%}
    Exclamations  : {r.get('exclamation_count', 0)}
    Review Text   : {str(r.get('body',''))[:80]}...
    Burst Activity: {flags.get('burst_activity', False)}
    All Extreme   : {flags.get('all_extreme_ratings', False)}
    Unverified %  : {flags.get('unverified_ratio', 0):.0%}
    Total Reviews : {flags.get('total_reviews', 0)}
""")

# ── Reviewer frequency analysis ───────────────────────
print(f"\n{'='*55}")
print("REVIEWER FREQUENCY ANALYSIS (Top 5 Suspicious):")
print("-" * 55)

reviewer_map = defaultdict(list)
for r in reviews:
    reviewer_map[r.get("reviewer_id", "UNKNOWN")].append(r)

suspicious = []
for rid, rlist in reviewer_map.items():
    ratings     = [r.get("rating", 3) for r in rlist]
    avg         = sum(ratings) / len(ratings)
    all_extreme = all(r in [1.0, 5.0] for r in ratings)
    avg_fraud   = sum(r.get("reviewer_fraud_score", 0) for r in rlist) / len(rlist)
    unverified  = sum(1 for r in rlist if not r.get("verified_purchase", False))
    suspicious.append({
        "reviewer_id"       : rid,
        "review_count"      : len(rlist),
        "avg_rating"        : round(avg, 2),
        "all_extreme_ratings": all_extreme,
        "avg_fraud_score"   : round(avg_fraud, 2),
        "unverified_count"  : unverified,
    })

suspicious.sort(key=lambda x: -x["avg_fraud_score"])
for i, r in enumerate(suspicious[:5], 1):
    print(f"""
[{i}] Reviewer ID    : {r['reviewer_id']}
    Total Reviews  : {r['review_count']}
    Avg Rating     : {r['avg_rating']} ⭐
    All Extreme    : {r['all_extreme_ratings']}
    Unverified     : {r['unverified_count']}/{r['review_count']}
    Fraud Score    : {r['avg_fraud_score']}
""")

# ── Product rating deviation ──────────────────────────
print(f"\n{'='*55}")
print("PRODUCT RATING DEVIATION ANALYSIS:")
print("-" * 55)

product_map = defaultdict(list)
for r in reviews:
    product_map[r.get("product_asin", "UNKNOWN")].append(r)

product_stats = []
for asin, rlist in product_map.items():
    all_ratings     = [r.get("rating", 3) for r in rlist]
    fake_ratings    = [r.get("rating", 3) for r in rlist
                       if r.get("reviewer_fraud_score", 0) >= 0.5]
    genuine_ratings = [r.get("rating", 3) for r in rlist
                       if r.get("reviewer_fraud_score", 0) < 0.5]

    avg_all     = sum(all_ratings) / len(all_ratings)
    avg_fake    = sum(fake_ratings) / max(len(fake_ratings), 1)
    avg_genuine = sum(genuine_ratings) / max(len(genuine_ratings), 1)
    inflation   = avg_fake - avg_genuine

    product_stats.append({
        "asin"           : asin,
        "total"          : len(rlist),
        "fake_count"     : len(fake_ratings),
        "genuine_count"  : len(genuine_ratings),
        "overall_avg"    : round(avg_all, 2),
        "fake_avg"       : round(avg_fake, 2),
        "genuine_avg"    : round(avg_genuine, 2),
        "inflation"      : round(inflation, 2),
    })

product_stats.sort(key=lambda x: -abs(x["inflation"]))
for p in product_stats:
    bar_fake    = "█" * p["fake_count"]
    bar_genuine = "░" * p["genuine_count"]
    print(f"""
  Product ASIN  : {p['asin']}
  Total Reviews : {p['total']}  [{bar_fake}{bar_genuine}]
  Fake Count    : {p['fake_count']}
  Genuine Count : {p['genuine_count']}
  Overall Avg   : {p['overall_avg']} ⭐
  Genuine Avg   : {p['genuine_avg']} ⭐
  Fake Avg      : {p['fake_avg']} ⭐
  Rating Inflat.: +{p['inflation']} stars boosted by fake reviews
""")

# ── Burst detection ───────────────────────────────────
print(f"\n{'='*55}")
print("BURST ACTIVITY DETECTION:")
print("-" * 55)

burst_reviewers = [
    r for r in suspicious
    if r["review_count"] >= 3 and r["avg_fraud_score"] >= 0.5
]
if burst_reviewers:
    for r in burst_reviewers[:5]:
        print(f"  ⚠️  Reviewer {r['reviewer_id']} posted "
              f"{r['review_count']} reviews "
              f"(fraud score: {r['avg_fraud_score']})")
else:
    print("  No burst activity detected.")

# ── Feature summary ───────────────────────────────────
print(f"\n{'='*55}")
print("FEATURE ENGINEERING SUMMARY (first 5 reviews):")
print("-" * 55)
for r in reviews[:5]:
    print(f"""
  Review  : {str(r.get('body',''))[:60]}...
  Features:
    word_count       = {r.get('word_count', 0)}
    caps_ratio       = {r.get('caps_ratio', 0):.3f}
    exclamations     = {r.get('exclamation_count', 0)}
    has_url          = {r.get('has_url', False)}
    spam_flag        = {r.get('spam_flag', False)}
    fraud_score      = {r.get('reviewer_fraud_score', 0):.3f}
    label            = {'🔴 FAKE' if r.get('reviewer_fraud_score',0) >= 0.5 else '🟢 GENUINE'}
""")

print(f"{'='*55}")
print("  ✅ All outputs generated successfully!")
print(f"{'='*55}\n")