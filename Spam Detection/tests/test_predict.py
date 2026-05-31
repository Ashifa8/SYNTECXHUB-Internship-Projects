"""
tests/test_predict.py
---------------------
Unit tests for the preprocessing and prediction modules.

Run:
    pytest tests/
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import clean_text


# ── Preprocessing tests ────────────────────────────────────────────────────────

class TestCleanText:
    def test_lowercase(self):
        assert clean_text("HELLO WORLD") == clean_text("hello world")

    def test_removes_url(self):
        result = clean_text("Visit http://win.com for free stuff")
        assert "http" not in result
        assert "win.com" not in result

    def test_removes_email(self):
        result = clean_text("Email us at promo@spam.com")
        assert "@" not in result

    def test_removes_numbers_and_punctuation(self):
        result = clean_text("Call 0800-123-456 now!!!")
        assert "0800" not in result
        assert "!" not in result

    def test_returns_string(self):
        assert isinstance(clean_text("Hello there"), str)

    def test_empty_string(self):
        result = clean_text("")
        assert result == ""

    def test_only_stopwords(self):
        # "the a is" are all stopwords — should produce empty or very short result
        result = clean_text("the a is")
        assert len(result.split()) == 0

    def test_spam_keywords_survive(self):
        result = clean_text("FREE prize winner cash")
        for word in ["free", "prize", "winner", "cash"]:
            assert word in result


# ── Prediction tests (skip if model not trained) ───────────────────────────────

try:
    from predict import predict
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model not trained yet. Run src/train.py first.")
class TestPredict:
    SPAM_MESSAGES = [
        "Congratulations you won a FREE iPhone click now",
        "URGENT your account suspended verify now",
        "FREE entry WIN 1000 cash text WIN to 87121",
    ]
    HAM_MESSAGES = [
        "Hey are we still meeting tomorrow at 1pm",
        "Can you send me the notes from class",
        "I will be late tonight, don't wait for me",
    ]

    def test_output_columns(self):
        result = predict(self.SPAM_MESSAGES[0])
        assert set(result.columns) == {"message", "prediction", "spam_prob", "ham_prob"}

    def test_single_string_input(self):
        result = predict("Hello world")
        assert len(result) == 1

    def test_list_input(self):
        result = predict(self.HAM_MESSAGES)
        assert len(result) == len(self.HAM_MESSAGES)

    def test_spam_detected(self):
        result = predict(self.SPAM_MESSAGES)
        assert (result["prediction"] == "SPAM").all(), "Expected all spam messages to be classified as SPAM"

    def test_ham_detected(self):
        result = predict(self.HAM_MESSAGES)
        assert (result["prediction"] == "HAM").all(), "Expected all ham messages to be classified as HAM"

    def test_probabilities_sum_to_100(self):
        result = predict(self.SPAM_MESSAGES + self.HAM_MESSAGES)
        sums = result["spam_prob"] + result["ham_prob"]
        assert (sums.round(0) == 100).all()

    def test_threshold_effect(self):
        msg = "You might have won something"
        low_thresh = predict(msg, threshold=0.01)
        high_thresh = predict(msg, threshold=0.99)
        # At very low threshold everything is SPAM; at very high threshold everything is HAM
        assert low_thresh["prediction"].iloc[0] == "SPAM"
        assert high_thresh["prediction"].iloc[0] == "HAM"
