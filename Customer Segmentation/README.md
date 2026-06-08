# Customer Segmentation using K-Means Clustering

A complete end-to-end machine learning project that segments mall customers into distinct groups using K-Means clustering. Built and executed on Kaggle with dual Tesla T4 GPUs.

## Project Overview

This project applies unsupervised machine learning to a retail customer dataset to identify meaningful customer segments. Each segment is then profiled and mapped to actionable marketing strategies.

## Dataset

**Mall Customer Segmentation Dataset** by Vijay Choudhary  
Source: [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

| Feature | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Gender | Male / Female |
| Age | Age of the customer |
| Annual_Income | Annual income in thousand dollars |
| Spending_Score | Mall-assigned score from 1 to 100 |

- 200 rows, 5 features
- No missing values

## Results

**Optimal K = 5** (determined via Elbow Method + Silhouette Score)  
**Silhouette Score = 0.4166**

| Cluster | Segment Name | Size | Avg Age | Avg Income | Avg Spending |
|---|---|---|---|---|---|
| 0 | Low Income Low Spenders | 20 (10%) | 46 | 27k$ | 18/100 |
| 1 | Young Active Spenders | 54 (27%) | 25 | 41k$ | 62/100 |
| 2 | High Value Premium Customers | 40 (20%) | 33 | 86k$ | 82/100 |
| 3 | Cautious High Earners | 39 (19.5%) | 40 | 86k$ | 19/100 |
| 4 | Average Middle Segment | 47 (23.5%) | 56 | 54k$ | 49/100 |

## Marketing Strategies per Segment

**Cluster 0 — Low Income Low Spenders**  
Older, budget-conscious customers. Target with value deals, discount loyalty cards, and affordable product lines.

**Cluster 1 — Young Active Spenders**  
Young customers spending above their income level. Target with trendy products, EMI/BNPL options, social media campaigns, and BOGO offers.

**Cluster 2 — High Value Premium Customers**  
Best segment. High income and high willingness to spend. Target with VIP loyalty programs, premium upsells, and exclusive events.

**Cluster 3 — Cautious High Earners**  
High income but low spending. Hard to convert. Focus on trust-building content, quality guarantees, customer reviews, and targeted flash sales.

**Cluster 4 — Average Middle Segment**  
Stable, moderate group. Largest segment after Cluster 1. Target with referral programs, seasonal discounts, loyalty points, and newsletters.

## Project Structure

```
customer-segmentation/
│
├── notebooks/
│   └── customer-segmentation.ipynb    # Main Kaggle notebook
│
├── data/
│   └── Mall_Customers.csv             # Raw dataset
│
├── outputs/
│   ├── customers_with_clusters.csv    # Dataset with cluster labels
│   └── cluster_summary.csv           # Per-cluster aggregate stats
│
├── images/
│   ├── eda_overview.png              # EDA plots
│   ├── elbow_silhouette.png          # Elbow + silhouette curves
│   ├── clusters_2d.png               # 2D cluster scatter plots
│   ├── clusters_pca.png              # PCA projection
│   └── cluster_boxplots.png          # Feature distributions per cluster
│
├── requirements.txt
└── README.md
```

## How to Run

**On Kaggle (recommended):**

1. Go to [Kaggle](https://www.kaggle.com) and create a new notebook
2. Under Settings, set Accelerator to GPU T4 x2
3. Enable Internet access
4. Add dataset: search "Mall Customer Segmentation" by vjchoudhary7
5. Copy the notebook from `notebooks/customer-segmentation.ipynb`
6. Run all cells in order

**Locally:**

```bash
git clone https://github.com/YOUR_USERNAME/customer-segmentation.git
cd customer-segmentation
pip install -r requirements.txt
jupyter notebook notebooks/customer-segmentation.ipynb
```

Note: Update the dataset path in Cell 4 from `/kaggle/working/` to your local path.

## Tech Stack

- Python 3.12
- scikit-learn — KMeans, StandardScaler, PCA, silhouette_score
- pandas, numpy — data manipulation
- matplotlib, seaborn — static visualizations
- plotly — interactive 3D cluster plot
- Kaggle — execution environment (Tesla T4 x2 GPU)

## Key Learnings

- Applied StandardScaler before clustering to normalize feature ranges
- Used Elbow Method and Silhouette Score together to validate K=5
- PCA projection confirmed clear cluster separation in 2D space
- Segment labels must be derived from actual cluster statistics, not assumed
