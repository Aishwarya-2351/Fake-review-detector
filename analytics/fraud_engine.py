# analytics/fraud_engine.py
from pymongo import MongoClient
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class FraudAnalyticsEngine:
    def __init__(self, mongo_uri="mongodb://localhost:27017"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client["reviews_db"]
        self.reviews = self.db["reviews"]
        self.reviewers = self.db["reviewers"]
        self.products = self.db["products"]

    def reviewer_frequency_analysis(self):
        """Flag reviewers posting too many reviews in a short window."""
        pipeline = [
            {"$group": {
                "_id": "$reviewer_id",
                "total_reviews": {"$sum": 1},
                "avg_rating": {"$avg": "$rating"},
                "products_reviewed": {"$addToSet": "$product_asin"},
                "dates": {"$push": "$scraped_at"},
                "fake_count": {"$sum": {"$cond": [
                    {"$gte": ["$fake_probability", 0.5]}, 1, 0
                ]}},
            }},
            {"$addFields": {
                "unique_products": {"$size": "$products_reviewed"},
                "fake_ratio": {
                    "$divide": ["$fake_count", "$total_reviews"]
                },
            }},
            {"$match": {"total_reviews": {"$gte": 3}}},
            {"$sort": {"fake_ratio": -1}},
        ]
        return list(self.reviews.aggregate(pipeline))

    def burst_detection(self, window_hours=24, threshold=5):
        """Detect unusual spikes in review volume for a product."""
        pipeline = [
            {"$group": {
                "_id": {
                    "product": "$product_asin",
                    "hour": {"$dateToString": {
                        "format": "%Y-%m-%d-%H",
                        "date": {"$toDate": "$scraped_at"}
                    }}
                },
                "count": {"$sum": 1},
                "avg_rating": {"$avg": "$rating"},
                "fake_count": {"$sum": {"$cond": [
                    {"$gte": ["$fake_probability", 0.5]}, 1, 0
                ]}},
            }},
            {"$match": {"count": {"$gte": threshold}}},
            {"$sort": {"count": -1}},
        ]
        return list(self.reviews.aggregate(pipeline))

    def rating_deviation_patterns(self):
        """Find products where fake reviews skew the average rating."""
        pipeline = [
            {"$group": {
                "_id": "$product_asin",
                "all_avg": {"$avg": "$rating"},
                "all_count": {"$sum": 1},
                "genuine_ratings": {"$push": {
                    "$cond": [
                        {"$lt": ["$fake_probability", 0.5]},
                        "$rating", None
                    ]
                }},
                "fake_ratings": {"$push": {
                    "$cond": [
                        {"$gte": ["$fake_probability", 0.5]},
                        "$rating", None
                    ]
                }},
            }},
        ]
        results = list(self.reviews.aggregate(pipeline))

        enriched = []
        for r in results:
            genuine = [x for x in r["genuine_ratings"] if x is not None]
            fake = [x for x in r["fake_ratings"] if x is not None]
            if genuine and fake:
                genuine_avg = statistics.mean(genuine)
                fake_avg = statistics.mean(fake)
                enriched.append({
                    "product_asin": r["_id"],
                    "overall_avg": round(r["all_avg"], 2),
                    "genuine_avg": round(genuine_avg, 2),
                    "fake_avg": round(fake_avg, 2),
                    "rating_inflation": round(fake_avg - genuine_avg, 2),
                    "fake_review_count": len(fake),
                    "genuine_review_count": len(genuine),
                })
        return sorted(enriched, key=lambda x: -abs(x["rating_inflation"]))

    def store_analytics_snapshot(self):
        """Run all analytics and store results in MongoDB."""
        snapshot = {
            "generated_at": datetime.utcnow(),
            "reviewer_frequency": self.reviewer_frequency_analysis()[:50],
            "burst_events": self.burst_detection(),
            "rating_deviations": self.rating_deviation_patterns()[:50],
        }
        self.db["analytics_snapshots"].insert_one(snapshot)
        return snapshot