# Online Fake Product Review Detection

A full-stack Big Data application that detects fraudulent Amazon product reviews using the Hadoop ecosystem and Machine Learning.

## Project Overview

E-commerce platforms face a serious challenge with fake reviews that manipulate consumer decisions. This project builds an end-to-end pipeline that scrapes Amazon product reviews, stores and processes data using Hadoop HDFS and MapReduce, performs sentiment analysis using Apache Hive, trains a Machine Learning classifier using Spark MLlib, stores verified results in MongoDB

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Collection | Python Scrapy |
| Storage | Hadoop HDFS |
| Processing | MapReduce |
| Querying | Apache Hive |
| Machine Learning | Spark MLlib / scikit-learn |
| Database | MongoDB |
| Frontend | React.js + Recharts |

## 🗂️ Dataset

Download from Kaggle and place CSV files in the project root:
https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 82% |
| Precision | 97% |
| F1 Score | 79% |
| AUC-ROC | 0.92 |
| Fake Reviews Detected | 1179 out of 5000 (23.58%) |
