"""
predict.py
----------
Load the trained pipeline and run predictions on new SMS messages.

Usage (Python):
    from src.predict import predict

    results = predict("Win a FREE iPhone now!", threshold=0.3)
    print(results)

Usage (CLI):
    python src/predict.py "Win a FREE iPhone now!"
"""

import pickle
import sys
from pathlib import Path
from typing import Union

import pandas as pd

from preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "spam_pipeline.pkl"


def load_model():
    """Load the serialized pipeline from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Run `python src/train.py` first to train and save the model."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(
    messages: Union[str, list],
    threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Predict whether one or more SMS messages are spam or ham.

    Parameters
    ----------
    messages : str or list of str
        Single message or a list of messages.
    threshold : float, optional
        Probability threshold for classifying a message as SPAM.
        Default 0.3 (errs toward catching spam).

    Returns
    -------
    pd.DataFrame
        Columns: message, prediction, spam_prob (%), ham_prob (%)
    """
    model = load_model()

    if isinstance(messages, str):
        messages = [messages]

    cleaned = [clean_text(m) for m in messages]
    probs = model.predict_proba(cleaned)
    preds = ["SPAM" if p[1] >= threshold else "HAM" for p in probs]

    return pd.DataFrame({
        "message":    messages,
        "prediction": preds,
        "spam_prob":  [round(p[1] * 100, 1) for p in probs],
        "ham_prob":   [round(p[0] * 100, 1) for p in probs],
    })


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"<message>\" [<message2> ...]")
        sys.exit(1)

    results = predict(sys.argv[1:])
    print(results.to_string(index=False))
