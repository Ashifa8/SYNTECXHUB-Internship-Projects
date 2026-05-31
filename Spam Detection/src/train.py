"""
train.py
--------
Download data, train & evaluate multiple models, tune the best one,
and save the final pipeline to models/spam_pipeline.pkl.

Run:
    python src/train.py
"""

import os
import json
import time
import pickle
import urllib.request
import zipfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, train_test_split,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import clean_text

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

SMS_PATH = DATA_DIR / "SMSSpamCollection"
MODEL_PATH = MODELS_DIR / "spam_pipeline.pkl"
META_PATH = MODELS_DIR / "model_meta.json"


# ── 1. Download data ───────────────────────────────────────────────────────────
def download_data() -> pd.DataFrame:
    if not SMS_PATH.exists():
        print("Downloading SMS Spam Collection dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        zip_path = DATA_DIR / "sms.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        os.remove(zip_path)
        print("Download complete.")
    else:
        print("Dataset already present.")

    df = pd.read_csv(SMS_PATH, sep="\t", header=None, names=["label", "message"])
    print(f"Loaded {len(df)} rows — spam: {(df['label']=='spam').sum()}, ham: {(df['label']=='ham').sum()}")
    return df


# ── 2. Preprocess ──────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    df = df.copy()
    df["clean_message"] = df["message"].apply(clean_text)
    df["label_encoded"] = (df["label"] == "spam").astype(int)
    return df


# ── 3. Benchmark models ────────────────────────────────────────────────────────
def evaluate_model(model, X_tr, X_te, y_tr, y_te, name, vec_name):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    elapsed = round(time.time() - t0, 3)
    y_pred = model.predict(X_te)
    return {
        "Model":     name,
        "Vectorizer": vec_name,
        "Accuracy":  round(accuracy_score(y_te, y_pred) * 100, 2),
        "Precision": round(precision_score(y_te, y_pred) * 100, 2),
        "Recall":    round(recall_score(y_te, y_pred) * 100, 2),
        "F1":        round(f1_score(y_te, y_pred) * 100, 2),
        "Time(s)":   elapsed,
    }


def benchmark(X_train, X_test, y_train, y_test):
    count_vec = CountVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    tfidf_vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)

    Xtr_cnt = count_vec.fit_transform(X_train)
    Xte_cnt = count_vec.transform(X_test)
    Xtr_tfidf = tfidf_vec.fit_transform(X_train)
    Xte_tfidf = tfidf_vec.transform(X_test)

    configs = [
        (MultinomialNB(alpha=0.1),                              "MultinomialNB",      "CountVec", Xtr_cnt,   Xte_cnt),
        (ComplementNB(alpha=0.1),                               "ComplementNB",       "CountVec", Xtr_cnt,   Xte_cnt),
        (MultinomialNB(alpha=0.1),                              "MultinomialNB",      "TF-IDF",   Xtr_tfidf, Xte_tfidf),
        (ComplementNB(alpha=0.1),                               "ComplementNB",       "TF-IDF",   Xtr_tfidf, Xte_tfidf),
        (LogisticRegression(C=1.0, max_iter=1000, random_state=42), "LogisticRegression", "TF-IDF", Xtr_tfidf, Xte_tfidf),
        (LinearSVC(C=1.0, random_state=42),                    "LinearSVC",          "TF-IDF",   Xtr_tfidf, Xte_tfidf),
    ]

    rows = []
    for model, name, vec_name, X_tr, X_te in configs:
        row = evaluate_model(model, X_tr, X_te, y_train, y_test, name, vec_name)
        rows.append(row)
        print(f"  {name:<22} ({vec_name})  F1={row['F1']}%")

    results = pd.DataFrame(rows).sort_values("F1", ascending=False)
    return results


# ── 4. Tune best pipeline ──────────────────────────────────────────────────────
def tune_pipeline(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        ("clf",   LogisticRegression(random_state=42, max_iter=1000)),
    ])

    param_grid = {
        "tfidf__max_features": [8000, 15000],
        "tfidf__ngram_range":  [(1, 1), (1, 2)],
        "clf__C":              [0.1, 1.0, 10.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"\nBest params : {grid.best_params_}")
    print(f"Best CV F1  : {round(grid.best_score_ * 100, 2)}%")

    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    print("\n" + classification_report(y_test, y_pred, target_names=["HAM", "SPAM"]))

    return best, grid


# ── 5. Save artifacts ──────────────────────────────────────────────────────────
def save_artifacts(pipeline, grid, y_test, y_pred):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved → {MODEL_PATH}")

    meta = {
        "model":       "LogisticRegression + TF-IDF",
        "created":     datetime.now().isoformat(),
        "best_params": grid.best_params_,
        "f1_cv":       round(grid.best_score_ * 100, 2),
        "f1_test":     round(f1_score(y_test, y_pred) * 100, 2),
        "precision":   round(precision_score(y_test, y_pred) * 100, 2),
        "recall":      round(recall_score(y_test, y_pred) * 100, 2),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"Metadata saved → {META_PATH}")
    return meta


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SMS Spam Detection — Training Pipeline")
    print("=" * 60)

    df = download_data()
    df = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"], df["label_encoded"],
        test_size=0.2, random_state=42, stratify=df["label_encoded"],
    )

    print("\n--- Benchmarking models ---")
    results = benchmark(X_train, X_test, y_train, y_test)
    print("\n" + results.to_string(index=False))

    print("\n--- Hyperparameter Tuning ---")
    best_pipeline, grid = tune_pipeline(X_train, X_test, y_train, y_test)

    y_pred_tuned = best_pipeline.predict(X_test)
    meta = save_artifacts(best_pipeline, grid, y_test, y_pred_tuned)

    print("\nFinal metrics:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print("\nTraining complete ✓")
