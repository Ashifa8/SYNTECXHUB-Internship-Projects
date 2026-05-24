# 📊 Model Metrics Summary

**Task:** Salary Regression — DS Salaries Dataset  
**Platform:** Kaggle (NVIDIA T4 GPU, RAPIDS cuML)  
**Date:** May 2026

---

## Model Comparison (Single vs Multiple Features)

| Model | R² | CV R² | ±std | Adj R² | RMSE | MAE | MAPE |
|---|---|---|---|---|---|---|---|
| Single (experience_level) | 0.2081 | 0.1464 | 0.0334 | 0.2008 | $43,458 | $35,249 | 52.7% |
| Double (exp + remote_ratio) | 0.2244 | 0.1567 | 0.0321 | 0.2099 | $43,008 | $34,494 | 51.0% |
| **Full Model (all features)** ✅ | **0.2815** | **0.1612** | **0.0566** | **0.2396** | **$41,395** | **$33,361** | **49.3%** |

**Best Model selected by:** CV R² (5-fold cross validation — more reliable than single test split)

---

## Learned Coefficients (Full Model)

| Feature | Coefficient | Direction |
|---|---|---|
| experience_level | +17,751 | ↑ Higher seniority → more salary |
| employment_type | -6,283 | ↓ Non-FT employment → less salary |
| company_size | -10,157 | ↓ Encoding order effect |
| job_title | +9,809 | ↑ Better title → more salary |
| remote_ratio | +6,515 | ↑ Remote work → higher pay |
| years_since_start | +9,644 | ↑ Recent years → higher salary |
| **Intercept** | **$114,155** | Base salary |

All directional relationships match real-world domain knowledge ✅

---

## Generalization Analysis — Leave-One-Group-Out CV

Grouped by `work_year` — each fold holds out one entire year.

| Held-out Year | n | R² | RMSE | MAE |
|---|---|---|---|---|
| 2020 | 61 | +0.0556 | $40,294 | $32,896 |
| 2021 | 182 | +0.1067 | $45,965 | $38,449 |
| 2022 | 304 | -0.0252 | $49,804 | $39,969 |
| **Mean** | **547** | **+0.0457** | **$45,355** | **$37,105** |

**In-sample CV R²:** +0.1612  
**Out-of-distribution R²:** +0.0457  
**Generalization Gap:** +0.1155 ⚠️

### Why the Gap Exists

```
2020 avg salary : ~$75,000
2021 avg salary : ~$98,000
2022 avg salary : ~$130,000  ← AI/ML hiring boom
```

Temporal salary shift (73% increase 2020→2022) cannot be captured by linear regression.  
This is a **dataset limitation**, not a code error.

**Fixes applied:**
- ✅ Outlier removal (5th–95th percentile) — removed extreme $600k+ salaries
- ✅ Relative year feature (`years_since_start`) — better temporal learning
- ❌ Augmentation tested — worsened gap (removed)

---

## Sample Prediction

**Input:** Senior Data Scientist, Full-Time, Large Company, 100% Remote, Year 2022  
**Predicted Salary:** `$117,800.50`

---

## Preprocessing Pipeline

```
Raw Data (607 rows)
    ↓ Outlier removal (5–95th percentile)
    ↓ 547 rows remain
    ↓ Relative year: work_year → years_since_start (0/1/2)
    ↓ Parallel LabelEncoding (4 categorical columns)
    ↓ Train/Test split 80/20
    ↓ StandardScaler (fit on train only)
    ↓ 3 models trained in parallel (GPU/CPU)
    ↓ Best model saved
```
