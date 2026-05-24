#  Data Science Salary Regression — Internship Task 1

> **Objective:** Build a Linear Regression model to predict Data Science salaries using experience, job title, employment type, and other features. Compare single vs multiple feature models and evaluate using RMSE and R².

---

##  Project Structure

```
salary-regression/
│
├── notebooks/
│   └── salary_regression.ipynb     ← Main Kaggle notebook (all cells)
│
├── src/
│   └── predict.py                  ← Inference script (load model & predict)
│
├── Models/                         ← Saved after running notebook
│   ├── best_model.pkl              ← Trained LinearRegression model
│   ├── scaler.pkl                  ← StandardScaler
│   └── label_encoders.pkl          ← LabelEncoders for categorical features
│
├── results/
│   └── metrics_summary.md          ← Final model comparison table
│
├── data/
│   └── README.md                   ← Dataset source info
│
├── requirements.txt
└── README.md
```

---

##  Dataset

**Source:** [DS Salaries Dataset — Kaggle](https://www.kaggle.com/datasets/ashifaikram/ds-salaries)

| Column | Type | Description |
|---|---|---|
| `experience_level` | Categorical | EN / MI / SE / EX |
| `employment_type` | Categorical | FT / PT / CT / FL |
| `company_size` | Categorical | S / M / L |
| `job_title` | Categorical | 50+ unique titles |
| `remote_ratio` | Numerical | 0, 50, 100 |
| `work_year` | Numerical | 2020 / 2021 / 2022 |
| `salary_in_usd` | Target | Salary in USD |

---

##  What This Notebook Does

| Step | Description |
|---|---|
| **1. Setup** | GPU detection (RAPIDS cuML), parallel workers |
| **2. Load Data** | Real dataset or synthetic fallback |
| **3. EDA** | Shape, missing values, statistics |
| **4. Preprocessing** | Outlier removal (5–95%), relative year feature, parallel LabelEncoding |
| **5. Feature Sets** | 3 subsets for single vs multiple comparison |
| **6. Training** | 3 LinearRegression models trained in parallel (GPU if available) |
| **7. Visualization** | Salary distribution, Actual vs Predicted, R² comparison |
| **8. LOGO CV** | Leave-One-Group-Out cross validation (generalization test) |
| **9. Save** | Best model saved via joblib |
| **10. Inference** | Predict salary for new sample |

---

##  Results

| Model | R² | CV R² | Adj R² | RMSE | MAE | MAPE |
|---|---|---|---|---|---|---|
| Single (experience_level) | 0.2081 | 0.1464 | 0.2008 | $43,458 | $35,249 | 52.7% |
| Double (exp + remote_ratio) | 0.2244 | 0.1567 | 0.2099 | $43,008 | $34,494 | 51.0% |
| **Full Model (all features)** | **0.2815** | **0.1612** | **0.2396** | **$41,395** | **$33,361** | **49.3%** |

**Best Model:** Full Model (all features)

### Why R² is ~0.28?
Low R² is expected on this dataset. Salary is heavily influenced by `job_title` diversity (50+ titles) and geography — linear regression on 6 encoded features has a natural ceiling. The model still correctly learns directional relationships:
- Higher `experience_level` → higher salary ↑
- Recent `work_year` → higher salary ↑
- `job_title` (Data Scientist, ML Engineer) → higher salary ↑

---

## Generalization Analysis (LOGO CV)

| Held-out Year | n | R² | RMSE | MAE |
|---|---|---|---|---|
| 2020 | 61 | +0.0556 | $40,294 | $32,896 |
| 2021 | 182 | +0.1067 | $45,965 | $38,449 |
| 2022 | 304 | -0.0252 | $49,804 | $39,969 |

**Gap = 0.1155 ** — Expected due to temporal salary shift (AI/ML hiring boom 2020→2022 caused ~73% salary increase). This is a dataset limitation, not a code issue.

---

##  How to Run

### On Kaggle (Recommended)
1. Upload `notebooks/salary_regression.ipynb` to Kaggle
2. Add dataset: `ashifaikram/ds-salaries`
3. Enable GPU accelerator (T4 x2)
4. Run All Cells

### Locally (CPU mode)
```bash
pip install -r requirements.txt
jupyter notebook notebooks/salary_regression.ipynb
```

### Inference Only
```bash
python src/predict.py
```

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| scikit-learn | LinearRegression, preprocessing, metrics |
| RAPIDS cuML | GPU-accelerated LinearRegression (Kaggle) |
| joblib | Parallel preprocessing + model saving |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualization |

---

##  Key Decisions

- **No precision/recall** — these are classification metrics, invalid for regression
- **Outlier removal at 5–95th percentile** — removes extreme salaries ($600k+) that distort the model
- **Relative year feature** — `years_since_start` (0/1/2) instead of raw `work_year` (2020/2021/2022) for better linear learning
- **CV R² for model selection** — more reliable than single test split R²
- **Augmentation tested but removed** — synthetic rows inflated CV R² without improving out-of-distribution performance

---

##  Author
Ashifa Ikram

**Internship Task 1** — Linear Regression  
*Dataset: DS Salaries (Kaggle) | Platform: Kaggle Notebooks (GPU)*

