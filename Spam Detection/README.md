# 📱 SMS Spam Detection

A machine learning project that classifies SMS messages as **spam** or **ham (legitimate)** using NLP techniques and multiple classification algorithms.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Results](#models--results)
- [Pipeline](#pipeline)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project builds an end-to-end SMS spam detection system using the **UCI SMS Spam Collection** dataset. It covers:

- **Exploratory Data Analysis (EDA)** with visualizations
- **Text preprocessing** using NLTK (lemmatization, stopword removal)
- **Feature extraction** using Bag-of-Words (CountVectorizer) and TF-IDF
- **Model comparison** across 6 classifier/vectorizer combinations
- **Hyperparameter tuning** with GridSearchCV
- **Model serialization** for deployment

---

## Dataset

**UCI SMS Spam Collection**  
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- Size: 5,572 SMS messages
- Classes: `ham` (87.4%) and `spam` (12.6%)

> The dataset is automatically downloaded when you run the notebook or training script.

---

## Project Structure

```
spam-detection-sms/
│
├── 📓 notebooks/
│   └── spam_detection.ipynb       # Full EDA + training walkthrough
│
├── 🐍 src/
│   ├── __init__.py
│   ├── preprocess.py              # Text cleaning & preprocessing
│   ├── train.py                   # Model training & evaluation
│   └── predict.py                 # Inference / prediction API
│
├── 🤖 models/
│   ├── spam_pipeline.pkl          # Trained model pipeline (generated)
│   └── model_meta.json            # Model metadata & metrics (generated)
│
├── 📊 data/
│   └── .gitkeep                   # Data downloaded automatically at runtime
│
├── 🧪 tests/
│   ├── __init__.py
│   └── test_predict.py            # Unit tests for prediction logic
│
├── 📄 docs/
│   └── results.md                 # Detailed results & analysis
│
├── requirements.txt               # Python dependencies
├── .gitignore                     # Files to exclude from git
└── README.md                      # You are here
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/spam-detection-sms.git
cd spam-detection-sms
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Option A — Run the Jupyter Notebook

```bash
jupyter notebook notebooks/spam_detection.ipynb
```

This walks through the full pipeline: EDA → preprocessing → training → evaluation.

### Option B — Train from the command line

```bash
python src/train.py
```

This downloads the dataset, trains all models, runs GridSearchCV, and saves the best pipeline to `models/spam_pipeline.pkl`.

### Option C — Run predictions

```python
from src.predict import predict

messages = [
    "Congratulations! You won a FREE iPhone. Click now!",
    "Hey, are we still meeting tomorrow at 1pm?",
    "URGENT: Your account is suspended. Verify now.",
]

results = predict(messages, threshold=0.3)
print(results)
```

```
                                             message prediction  spam_prob  ham_prob
0  Congratulations! You won a FREE iPhone. Click...       SPAM       98.5       1.5
1        Hey, are we still meeting tomorrow at 1pm?        HAM        1.2      98.8
2       URGENT: Your account is suspended. Verify...       SPAM       97.1       2.9
```

---

## Models & Results

Six model + vectorizer combinations were benchmarked:

| Model               | Vectorizer  | Accuracy | Precision | Recall  | F1 Score |
|---------------------|-------------|----------|-----------|---------|----------|
| LinearSVC           | TF-IDF      | ~98.8%   | ~97.5%    | ~97.0%  | ~97.2%   |
| LogisticRegression  | TF-IDF      | ~98.6%   | ~97.2%    | ~96.5%  | ~96.8%   |
| ComplementNB        | TF-IDF      | ~97.9%   | ~95.8%    | ~95.3%  | ~95.5%   |
| MultinomialNB       | TF-IDF      | ~97.6%   | ~95.1%    | ~94.9%  | ~95.0%   |
| ComplementNB        | CountVec    | ~97.4%   | ~94.7%    | ~94.2%  | ~94.4%   |
| MultinomialNB       | CountVec    | ~97.1%   | ~94.3%    | ~93.8%  | ~94.0%   |

> **Best model after tuning:** `LogisticRegression + TF-IDF` with GridSearchCV  
> Best params: `ngram_range=(1,2)`, `max_features=15000`, `C=1.0`

See [`docs/results.md`](docs/results.md) for detailed analysis.

---

## Pipeline

```
Raw SMS Text
     │
     ▼
Text Cleaning
  • Lowercase
  • Remove URLs, emails, numbers, punctuation
  • Lemmatization (NLTK WordNetLemmatizer)
  • Remove stopwords
     │
     ▼
TF-IDF Vectorization
  • max_features=15,000
  • ngram_range=(1,2) — unigrams + bigrams
  • sublinear_tf=True
     │
     ▼
Logistic Regression Classifier
  • C=1.0
  • Threshold=0.3 (adjustable for recall/precision trade-off)
     │
     ▼
Prediction: HAM / SPAM + Confidence Score
```

---

## Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

This project is licensed under the [MIT License](LICENSE).

---

*Built with ❤️ using Python, scikit-learn, and NLTK*
