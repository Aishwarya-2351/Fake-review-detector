#!/bin/bash
# hadoop/mapreduce/run_mapreduce.sh

INPUT="/user/reviews/raw/raw_reviews.json"
OUTPUT="/user/reviews/mapreduce_output"

# Remove previous output if exists
hdfs dfs -rm -r -f $OUTPUT

hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
  -files spam_filter_mapper.py,spam_filter_reducer.py \
  -mapper "python3 spam_filter_mapper.py" \
  -reducer "python3 spam_filter_reducer.py" \
  -input $INPUT \
  -output $OUTPUT

echo "MapReduce done. Merging output..."
hdfs dfs -getmerge $OUTPUT/part-* ../processed_reviews.json
echo "Merged output saved locally as processed_reviews.json"