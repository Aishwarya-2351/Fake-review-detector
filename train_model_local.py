# train_model_local.py
import json
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pickle
import os

# ── 1. Load processed reviews ─────────────────────────
print("\n" + "="*55)
print("  FAKE REVIEW DETECTOR — ML TRAINING")
print("="*55)

print("\n[1/7] Loading processed reviews...")
reviews = []
with open("hadoop/map/processed_reviews.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            reviews.append(json.loads(line))

print(f"      Loaded {len(reviews)} reviews")

# ── 2. Feature engineering ────────────────────────────
print("\n[2/7] Engineering features...")

SPAM_REGEX = re.compile(
    r"\b(buy now|click here|visit our|free|discount|promo|coupon)\b",
    re.IGNORECASE
)
POSITIVE_REGEX = re.compile(
    r"\b(great|excellent|amazing|perfect|love|best|awesome|fantastic)\b",
    re.IGNORECASE
)
NEGATIVE_REGEX = re.compile(
    r"\b(terrible|awful|horrible|worst|hate|broken|useless|scam)\b",
    re.IGNORECASE
)

def extract_features(review):
    body   = review.get("body", "")
    rating = float(review.get("rating", 3.0))
    flags  = review.get("reviewer_flags", {})

    word_count         = len(body.split())
    caps_ratio         = sum(1 for c in body if c.isupper()) / max(len(body), 1)
    exclamation_count  = body.count("!")
    has_url            = int(bool(re.search(r"https?://", body)))
    spam_word_hit      = int(bool(SPAM_REGEX.search(body)))
    positive_signal    = int(bool(POSITIVE_REGEX.search(body)))
    negative_signal    = int(bool(NEGATIVE_REGEX.search(body)))
    sentiment_mismatch = int(rating >= 4.0 and bool(NEGATIVE_REGEX.search(body)))
    all_extreme        = int(flags.get("all_extreme_ratings", False))
    burst_activity     = int(flags.get("burst_activity", False))
    unverified_ratio   = float(flags.get("unverified_ratio", 0.0))
    total_reviews      = int(flags.get("total_reviews", 1))
    verified           = int(review.get("verified_purchase", False))

    return {
        "body"              : body,
        "word_count"        : word_count,
        "caps_ratio"        : round(caps_ratio, 4),
        "exclamation_count" : exclamation_count,
        "has_url"           : has_url,
        "spam_word_hit"     : spam_word_hit,
        "positive_signal"   : positive_signal,
        "negative_signal"   : negative_signal,
        "sentiment_mismatch": sentiment_mismatch,
        "all_extreme"       : all_extreme,
        "burst_activity"    : burst_activity,
        "unverified_ratio"  : unverified_ratio,
        "total_reviews"     : total_reviews,
        "verified_purchase" : verified,
        "rating"            : rating,
    }

feature_list = [extract_features(r) for r in reviews]
df = pd.DataFrame(feature_list)

# ── Honest labeling — use ground truth + 5% noise ────
df["label"] = [int(r.get("is_fake_ground_truth", False)) for r in reviews]

# Add 5% label noise so model cannot memorize perfectly
random.seed(42)
noise_idx = random.sample(range(len(df)), max(1, int(len(df) * 0.05)))
for idx in noise_idx:
    df.at[idx, "label"] = 1 - df.at[idx, "label"]

print(f"      Features extracted for {len(df)} reviews")
print(f"      Fake   : {df['label'].sum()}")
print(f"      Genuine: {(df['label'] == 0).sum()}")

# ── 3. Prepare data ───────────────────────────────────
print("\n[3/7] Preparing training data...")

NUMERIC_FEATURES = [
    "word_count", "caps_ratio", "exclamation_count",
    "has_url", "spam_word_hit", "positive_signal",
    "negative_signal", "sentiment_mismatch", "all_extreme",
    "burst_activity", "unverified_ratio", "total_reviews",
    "verified_purchase", "rating"
]

X_numeric = df[NUMERIC_FEATURES].values
y         = df["label"].values
texts     = df["body"].values

# TF-IDF on review text
print("      Fitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer(
    max_features=300,
    ngram_range=(1, 2),
    min_df=2,
    stop_words="english"
)
X_tfidf = tfidf.fit_transform(texts).toarray()

# Combine TF-IDF + numeric features
X = np.hstack([X_tfidf, X_numeric])
print(f"      Feature matrix shape: {X.shape}")

# ── Handle class imbalance ────────────────────────────
fake_count    = int(y.sum())
genuine_count = int((y == 0).sum())
print(f"      Class balance — Genuine: {genuine_count}  Fake: {fake_count}")

if fake_count < genuine_count:
    # Oversample minority class
    X_genuine = X[y == 0]
    X_fake    = X[y == 1]
    X_fake_up = resample(
        X_fake,
        replace=True,
        n_samples=genuine_count,
        random_state=42
    )
    X_balanced = np.vstack([X_genuine, X_fake_up])
    y_balanced = np.array([0] * genuine_count + [1] * genuine_count)
elif genuine_count < fake_count:
    X_fake    = X[y == 1]
    X_genuine = X[y == 0]
    X_gen_up  = resample(
        X_genuine,
        replace=True,
        n_samples=fake_count,
        random_state=42
    )
    X_balanced = np.vstack([X_fake, X_gen_up])
    y_balanced = np.array([1] * fake_count + [0] * fake_count)
else:
    X_balanced = X
    y_balanced = y

print(f"      Balanced dataset size: {len(X_balanced)}")

# ── Train / test split ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced,
    test_size=0.2,
    random_state=42,
    stratify=y_balanced
)
print(f"      Train size : {len(X_train)}")
print(f"      Test size  : {len(X_test)}")

# ── 4. Train models ───────────────────────────────────
print("\n[4/7] Training models...")

models = {
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=80,
        max_depth=3,          # shallower tree = less overfitting
        learning_rate=0.1,
        min_samples_leaf=5,   # prevents memorizing small clusters
        subsample=0.8,        # stochastic boosting adds variance
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=80,
        max_depth=4,          # shallower = less overfitting
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),
}

results = {}
for name, model in models.items():
    print(f"\n      Training {name}...")
    model.fit(X_train, y_train)
    y_pred    = model.predict(X_test)
    y_prob    = model.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring="f1"
    )

    results[name] = {
        "model"     : model,
        "y_pred"    : y_pred,
        "y_prob"    : y_prob,
        "accuracy"  : round(accuracy_score(y_test, y_pred), 4),
        "precision" : round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall"    : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1"        : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc"   : round(roc_auc_score(y_test, y_prob), 4),
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std" : round(cv_scores.std(), 4),
    }
    print(f"      ✅ {name} trained")

# ── 5. Print metrics ──────────────────────────────────
print("\n[5/7] Model evaluation results...")
print("\n" + "="*55)
print("  MODEL PERFORMANCE COMPARISON")
print("="*55)

best_model_name = max(results, key=lambda x: results[x]["auc_roc"])

for name, res in results.items():
    tag = " ← BEST" if name == best_model_name else ""
    print(f"""
  {name}{tag}
  {'─'*45}
  Accuracy  : {res['accuracy']:.4f}  ({res['accuracy']*100:.2f}%)
  Precision : {res['precision']:.4f}
  Recall    : {res['recall']:.4f}
  F1 Score  : {res['f1']:.4f}
  AUC-ROC   : {res['auc_roc']:.4f}
  CV F1     : {res['cv_f1_mean']:.4f} ± {res['cv_f1_std']:.4f}
""")

print("\nDetailed Classification Report (Best Model):")
print("-"*55)
best = results[best_model_name]
print(classification_report(
    y_test, best["y_pred"],
    target_names=["Genuine", "Fake"]
))

# ── 6. Save model ─────────────────────────────────────
print("\n[6/7] Saving model...")
os.makedirs("models", exist_ok=True)

model_data = {
    "model"           : results[best_model_name]["model"],
    "tfidf_vectorizer": tfidf,
    "numeric_features": NUMERIC_FEATURES,
    "best_model_name" : best_model_name,
    "metrics"         : {
        k: v for k, v in results[best_model_name].items()
        if k not in ("model", "y_pred", "y_prob")
    },
}
with open("models/fake_review_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("      ✅ Model saved to models/fake_review_model.pkl")

# ── 7. Generate charts ────────────────────────────────
print("\n[7/7] Generating evaluation charts...")
os.makedirs("outputs", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Fake Review Detector — ML Evaluation",
    fontsize=16, fontweight="bold"
)

# Chart 1: Confusion matrix
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Reds",
    xticklabels=["Genuine", "Fake"],
    yticklabels=["Genuine", "Fake"],
    ax=axes[0, 0]
)
axes[0, 0].set_title(f"Confusion Matrix ({best_model_name})")
axes[0, 0].set_ylabel("Actual")
axes[0, 0].set_xlabel("Predicted")

# Chart 2: ROC curves
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    axes[0, 1].plot(
        fpr, tpr,
        label=f"{name} (AUC={res['auc_roc']})"
    )
axes[0, 1].plot([0, 1], [0, 1], "k--", label="Random baseline")
axes[0, 1].set_title("ROC Curve")
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Chart 3: Model comparison bar chart
metric_names  = ["accuracy", "precision", "recall", "f1", "auc_roc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
x     = np.arange(len(metric_names))
width = 0.35
colors = ["#3b82f6", "#ef4444"]
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metric_names]
    axes[1, 0].bar(
        x + i * width, vals, width,
        label=name, alpha=0.85,
        color=colors[i]
    )
axes[1, 0].set_title("Model Metrics Comparison")
axes[1, 0].set_xticks(x + width / 2)
axes[1, 0].set_xticklabels(metric_labels, fontsize=9)
axes[1, 0].set_ylim(0, 1.15)
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3, axis="y")
for i, (name, res) in enumerate(results.items()):
    for j, m in enumerate(metric_names):
        axes[1, 0].text(
            j + i * width, res[m] + 0.02,
            f"{res[m]:.2f}",
            ha="center", fontsize=7
        )

# Chart 4: Feature importance (numeric features only)
best_clf = results[best_model_name]["model"]
if hasattr(best_clf, "feature_importances_"):
    importances       = best_clf.feature_importances_
    n_tfidf           = X_tfidf.shape[1]
    numeric_imp       = importances[n_tfidf:]
    indices           = np.argsort(numeric_imp)[::-1]
    top_features      = [NUMERIC_FEATURES[i] for i in indices[:10]]
    top_importance    = numeric_imp[indices[:10]]
    bar_colors        = ["#ef4444" if v > 0.05 else "#3b82f6"
                         for v in top_importance[::-1]]
    axes[1, 1].barh(
        top_features[::-1],
        top_importance[::-1],
        color=bar_colors
    )
    axes[1, 1].set_title("Top Feature Importances")
    axes[1, 1].set_xlabel("Importance Score")
    axes[1, 1].grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(top_importance[::-1]):
        axes[1, 1].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8)

plt.tight_layout()
chart_path = "outputs/ml_evaluation.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
print(f"      ✅ Charts saved to {chart_path}")
plt.show()

print(f"""
{'='*55}
  ✅ TRAINING COMPLETE
{'='*55}
  Best Model : {best_model_name}
  AUC-ROC    : {results[best_model_name]['auc_roc']}
  F1 Score   : {results[best_model_name]['f1']}
  Accuracy   : {results[best_model_name]['accuracy']}
  Model File : models/fake_review_model.pkl
  Charts     : outputs/ml_evaluation.png
{'='*55}
""")