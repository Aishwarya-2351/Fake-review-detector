# store_to_mongodb.py
import json
from pymongo import MongoClient
from datetime import datetime

print("="*55)
print("  STORING REVIEWS TO MONGODB")
print("="*55)

# Connect to MongoDB
# If MongoDB is not installed, this uses mongomock
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    client.server_info()
    print("✅ Connected to MongoDB")
except Exception:
    print("⚠️  MongoDB not running — using mongomock (in-memory)")
    import mongomock
    client = mongomock.MongoClient()

db = client["reviews_db"]

# Load processed reviews
reviews = []
with open("hadoop/map/processed_reviews.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            reviews.append(json.loads(line))

# Clear existing
db.reviews.drop()

# Insert all reviews
result = db.reviews.insert_many(reviews)
print(f"✅ Inserted {len(result.inserted_ids)} reviews into MongoDB")

# Insert fraud summary
fake    = [r for r in reviews if r.get("reviewer_fraud_score", 0) >= 0.5]
genuine = [r for r in reviews if r.get("reviewer_fraud_score", 0) < 0.5]

summary = {
    "generated_at"   : datetime.utcnow().isoformat(),
    "total_reviews"  : len(reviews),
    "fake_reviews"   : len(fake),
    "genuine_reviews": len(genuine),
    "fake_percentage": round(len(fake) / max(len(reviews), 1) * 100, 2),
}
db.analytics_snapshots.insert_one(summary)
print(f"✅ Analytics snapshot stored")

# Verify
total   = db.reviews.count_documents({})
fakes   = db.reviews.count_documents({"reviewer_fraud_score": {"$gte": 0.5}})

print(f"""
{'='*55}
  MONGODB STORAGE COMPLETE
{'='*55}
  Database      : reviews_db
  Collection    : reviews
  Total docs    : {total}
  Fake reviews  : {fakes}
  Genuine       : {total - fakes}
{'='*55}
""")