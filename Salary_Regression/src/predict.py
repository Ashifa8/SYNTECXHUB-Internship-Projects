"""
predict.py — Load saved model and predict salary for a new sample.

Usage:
    python src/predict.py

Requirements:
    - models/best_model.pkl
    - models/scaler.pkl
    - models/label_encoders.pkl

    Run the notebook first to generate these files.
"""

import joblib
import numpy as np
import pandas as pd
import os

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENC_PATH    = os.path.join(BASE_DIR, "models", "label_encoders.pkl")


def load_artifacts():
    """Load model, scaler, and encoders from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Please run the notebook first to train and save the model."
        )
    model          = joblib.load(MODEL_PATH)
    scaler         = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENC_PATH)
    print("✅ Model, scaler, and encoders loaded successfully.")
    return model, scaler, label_encoders


def encode_sample(sample_dict: dict, label_encoders: dict) -> dict:
    """Encode categorical values using saved LabelEncoders."""
    encoded = {}
    for col, val in sample_dict.items():
        if col in label_encoders:
            le = label_encoders[col]
            if val not in le.classes_:
                print(f"  ⚠ '{val}' not seen during training for '{col}'. "
                      f"Using fallback: '{le.classes_[0]}'")
                val = le.classes_[0]
            encoded[col] = int(le.transform([val])[0])
        else:
            encoded[col] = val
    return encoded


def predict_salary(sample_dict: dict) -> float:
    """
    Predict salary for a given input sample.

    Args:
        sample_dict: dict with keys matching training features.
                     Categorical values should be raw strings (e.g. "SE", "FT").

    Returns:
        Predicted salary in USD (float).
    """
    model, scaler, label_encoders = load_artifacts()

    # Encode categoricals
    encoded = encode_sample(sample_dict, label_encoders)

    # Feature order must match training
    ALL_FEATURES = [
        "experience_level", "employment_type", "company_size",
        "job_title", "remote_ratio", "years_since_start"
    ]

    # Build dataframe
    row = {col: encoded.get(col, 0) for col in ALL_FEATURES}
    df_sample = pd.DataFrame([row])

    # Scale
    df_scaled = pd.DataFrame(
        scaler.transform(df_sample[ALL_FEATURES]),
        columns=ALL_FEATURES
    )

    # Predict
    pred = model.predict(df_scaled)[0]
    if hasattr(pred, "item"):
        pred = pred.item()

    return pred


# ── Example usage ─────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("  SALARY PREDICTION — DS Salaries Linear Regression")
    print("=" * 55)

    # Sample 1: Senior Data Scientist, Full Time, Large Company
    sample1 = {
        "experience_level":  "SE",            # Senior
        "employment_type":   "FT",            # Full Time
        "company_size":      "L",             # Large
        "job_title":         "Data Scientist",
        "remote_ratio":      100,             # Fully remote
        "years_since_start": 2,              # year 2022
    }

    # Sample 2: Entry level, Part Time, Small Company
    sample2 = {
        "experience_level":  "EN",            # Entry level
        "employment_type":   "PT",            # Part Time
        "company_size":      "S",             # Small
        "job_title":         "Data Analyst",
        "remote_ratio":      0,              # On-site
        "years_since_start": 0,             # year 2020
    }

    for i, sample in enumerate([sample1, sample2], 1):
        print(f"\nSample {i}:")
        for k, v in sample.items():
            print(f"  {k:<25} = {v}")

        try:
            salary = predict_salary(sample)
            print(f"\n  ➤ Predicted Salary : ${salary:,.2f}")
        except FileNotFoundError as e:
            print(f"\n  ❌ {e}")

    print("\n" + "=" * 55)
