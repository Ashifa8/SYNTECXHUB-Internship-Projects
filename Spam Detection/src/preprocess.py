"""
preprocess.py
-------------
Text cleaning and preprocessing utilities for SMS spam detection.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data on first use
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean and normalize a raw SMS message.

    Steps:
        1. Lowercase
        2. Remove URLs and email addresses
        3. Remove non-alphabetic characters
        4. Tokenize, lemmatize, and drop stopwords / single-char tokens

    Parameters
    ----------
    text : str
        Raw SMS message string.

    Returns
    -------
    str
        Cleaned, lemmatized text ready for vectorization.

    Examples
    --------
    >>> clean_text("FREE entry WIN £1000! Visit www.win.com NOW!!!")
    'free entry win visit'
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"\S+@\S+", "", text)           # remove emails
    text = re.sub(r"[^a-z\s]", "", text)          # keep only letters
    tokens = text.split()
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if len(t) > 1 and t not in _stop_words
    ]
    return " ".join(tokens)
