#!/bin/bash
# ============================================================
#  upload_to_hdfs.sh
#  Uploads scraped Amazon review data to Hadoop HDFS
#  Project: Online Fake Product Review Detection
# ============================================================

# ── Configuration ────────────────────────────────────────────
RAW_FILE="../../Scraper/raw_reviews.json"
PROCESSED_FILE="../map/processed_reviews.json"

HDFS_RAW_PATH="/user/reviews/raw/"
HDFS_PROCESSED_PATH="/user/reviews/processed/"
HDFS_OUTPUT_PATH="/user/reviews/mapreduce_output/"
HDFS_MODEL_PATH="/user/models/"

echo "============================================================"
echo "  ONLINE FAKE REVIEW DETECTOR — HDFS UPLOAD"
echo "============================================================"

# ── Step 1: Check HDFS is running ────────────────────────────
echo ""
echo "[1/6] Checking HDFS connection..."
hdfs dfs -ls / > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "  ERROR: HDFS is not running."
    echo "  Start Hadoop first using:"
    echo "    start-dfs.sh"
    echo "    start-yarn.sh"
    exit 1
fi
echo "  ✅ HDFS is running"

# ── Step 2: Check source files exist ─────────────────────────
echo ""
echo "[2/6] Checking source files..."

if [ ! -f "$RAW_FILE" ]; then
    echo "  ERROR: Raw reviews file not found at $RAW_FILE"
    echo "  Run: python amazon_scraper.py first"
    exit 1
fi
echo "  ✅ Raw reviews file found: $RAW_FILE"

if [ ! -f "$PROCESSED_FILE" ]; then
    echo "  WARNING: Processed file not found at $PROCESSED_FILE"
    echo "  Will upload raw file only"
fi

# ── Step 3: Create HDFS directories ──────────────────────────
echo ""
echo "[3/6] Creating HDFS directories..."

hdfs dfs -mkdir -p $HDFS_RAW_PATH
echo "  ✅ Created: $HDFS_RAW_PATH"

hdfs dfs -mkdir -p $HDFS_PROCESSED_PATH
echo "  ✅ Created: $HDFS_PROCESSED_PATH"

hdfs dfs -mkdir -p $HDFS_OUTPUT_PATH
echo "  ✅ Created: $HDFS_OUTPUT_PATH"

hdfs dfs -mkdir -p $HDFS_MODEL_PATH
echo "  ✅ Created: $HDFS_MODEL_PATH"

# ── Step 4: Upload raw reviews ────────────────────────────────
echo ""
echo "[4/6] Uploading raw reviews to HDFS..."

# Remove old file if exists
hdfs dfs -rm -f ${HDFS_RAW_PATH}raw_reviews.json > /dev/null 2>&1

# Upload fresh file
hdfs dfs -put $RAW_FILE ${HDFS_RAW_PATH}raw_reviews.json

if [ $? -eq 0 ]; then
    echo "  ✅ Uploaded: raw_reviews.json → $HDFS_RAW_PATH"
else
    echo "  ERROR: Failed to upload raw_reviews.json"
    exit 1
fi

# ── Step 5: Upload processed reviews (if exists) ─────────────
echo ""
echo "[5/6] Uploading processed reviews to HDFS..."

if [ -f "$PROCESSED_FILE" ]; then
    hdfs dfs -rm -f ${HDFS_PROCESSED_PATH}processed_reviews.json > /dev/null 2>&1
    hdfs dfs -put $PROCESSED_FILE ${HDFS_PROCESSED_PATH}processed_reviews.json

    if [ $? -eq 0 ]; then
        echo "  ✅ Uploaded: processed_reviews.json → $HDFS_PROCESSED_PATH"
    else
        echo "  WARNING: Failed to upload processed_reviews.json"
    fi
else
    echo "  SKIPPED: processed_reviews.json not found"
fi

# ── Step 6: Verify uploads ────────────────────────────────────
echo ""
echo "[6/6] Verifying uploads..."

echo ""
echo "  Contents of $HDFS_RAW_PATH:"
hdfs dfs -ls $HDFS_RAW_PATH

echo ""
echo "  Contents of $HDFS_PROCESSED_PATH:"
hdfs dfs -ls $HDFS_PROCESSED_PATH

echo ""
echo "  HDFS disk usage:"
hdfs dfs -du -h /user/reviews/

# ── Summary ───────────────────────────────────────────────────
RAW_SIZE=$(wc -c < "$RAW_FILE" 2>/dev/null || echo 0)
RAW_LINES=$(wc -l < "$RAW_FILE" 2>/dev/null || echo 0)

echo ""
echo "============================================================"
echo "  ✅ HDFS UPLOAD COMPLETE"
echo "============================================================"
echo "  Raw file size    : $(( RAW_SIZE / 1024 )) KB"
echo "  HDFS raw path    : $HDFS_RAW_PATH"
echo "  HDFS processed   : $HDFS_PROCESSED_PATH"
echo "  HDFS output      : $HDFS_OUTPUT_PATH"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "  1. Run MapReduce job:"
echo "     cd hadoop/map"
echo "     hadoop jar hadoop-streaming.jar \\"
echo "       -mapper spam_filter_mapper.py \\"
echo "       -reducer spam_filter_reducer.py \\"
echo "       -input $HDFS_RAW_PATH \\"
echo "       -output $HDFS_OUTPUT_PATH"
echo ""
echo "  2. Run Hive queries:"
echo "     hive -f ../../hive/create_tables.hql"
echo "     hive -f ../../hive/fraud_queries.hql"
echo "============================================================"