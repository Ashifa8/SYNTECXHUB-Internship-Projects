# Movie Recommendation System

A machine learning project that recommends movies using **Content-Based Filtering** and **Collaborative Filtering (SVD)** built on the TMDB 5000 dataset.

---

## Results

| Metric | Score |
|--------|-------|
| RMSE | 0.4055 |
| MAE | 0.3202 |
| Precision@5 | 0.8468 |
| Precision@10 | 0.6518 |

---

## Project Structure

```
movie-recommendation-system/
│
├── README.md                          # Project overview
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Files to ignore in git
│
├── notebook/
│   └── movie-recommendation-task-4.ipynb   # Full pipeline on Kaggle
│
├── app/
│   └── app.py                         # Flask REST API
│
└── outputs/
    └── (plots saved here after running notebook)
```

---

## How It Works

### Content-Based Filtering
Each movie is represented as a weighted text "soup":
- Director name repeated 3 times (highest weight)
- Top 5 cast members repeated 2 times
- Genres repeated 2 times
- Keywords once
- First 50 words of plot overview

TF-IDF vectorization is applied on this soup, then cosine similarity is computed across all 5000 movies.

### Collaborative Filtering
SVD (Singular Value Decomposition) from the `scikit-surprise` library is used. Ratings are synthetically generated from TMDB vote averages and vote counts to simulate user behavior.

### Hybrid Recommender
Both scores are combined using a weighted blend:

```
hybrid_score = alpha * content_score + (1 - alpha) * cf_score
```

Default alpha = 0.6 (60% content, 40% collaborative).

---

## Dataset

TMDB 5000 Movies dataset — add to Kaggle notebook before running:

https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Two files are used:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

---

## Setup and Usage

### Run the Notebook
1. Open Kaggle and create a new notebook
2. Add dataset: `tmdb-movie-metadata`
3. Enable GPU: Settings → Accelerator → GPU T4
4. Upload and run `notebook/movie-recommendation-task-4.ipynb`

### Run the Flask API Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
python app/app.py
```

Test the endpoint:

```bash
curl "http://localhost:5000/recommend?title=Inception&n=5"
```

Sample response:

```json
{
  "query": "Inception",
  "recommendations": [
    {
      "title": "Interstellar",
      "vote_average": 8.1,
      "similarity": 0.4231,
      "genres": "Science Fiction, Adventure",
      "director": "Christopher Nolan"
    }
  ]
}
```

Health check:

```bash
curl "http://localhost:5000/health"
```

---

## API Endpoints

| Endpoint | Method | Params | Description |
|----------|--------|--------|-------------|
| `/recommend` | GET | `title`, `n` | Get top-N similar movies |
| `/health` | GET | — | Check if API is running |

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| pandas, numpy | Data processing |
| scikit-learn | TF-IDF, cosine similarity |
| scikit-surprise | SVD collaborative filtering |
| matplotlib, plotly | EDA visualizations |
| wordcloud | Overview text visualization |
| flask | REST API |

---

## Notebook Cells Overview

| Cell | Description |
|------|-------------|
| 1 | Project title and overview (Markdown) |
| 2 | GPU check and library installs |
| 3 | All imports |
| 4 | Load TMDB movies and credits CSV |
| 5 | Null values and basic statistics |
| 6 | Feature distribution plots |
| 7 | Genre count and average rating by genre |
| 8 | WordCloud and top movies by weighted score |
| 9 | Merge credits, parse JSON fields, clean names |
| 10 | Build weighted metadata soup |
| 11 | TF-IDF matrix and cosine similarity |
| 12 | Content-based recommend function |
| 13 | Sample queries — Dark Knight, Inception, etc. |
| 14 | Generate synthetic ratings from TMDB votes |
| 15 | Train SVD with 5-fold cross-validation |
| 16 | CF predict function for a given user |
| 17 | Hybrid recommender (content + CF blend) |
| 18 | Evaluation — RMSE, MAE, Precision@K |
| 19 | Save Flask app as app.py |
| 20 | Save model artifacts as .pkl files |
