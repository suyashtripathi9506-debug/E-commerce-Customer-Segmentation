# Customer Segmentation & Recommendation (E-commerce / Marketing)

Techniques: RFM Analysis, K-Means Clustering, PCA, Recommendation System

## Files
| File | Purpose |
|---|---|
| `data_generator.py` | Creates a synthetic transaction dataset (or use your own CSV) |
| `rfm_analysis.py` | Cleaning, Recency/Frequency/Monetary, 1-5 scoring, rule-based segments |
| `clustering.py` | Log + standard scaling, elbow & silhouette, K-Means, cluster profiling/naming |
| `pca_visualization.py` | Explained variance, 2D cluster plot, loadings |
| `recommender.py` | Item-based collaborative filtering + segment-popularity hybrid + hit-rate evaluation |
| `main.py` | Runs everything end to end |

## Run
```
pip install -r requirements.txt
python main.py
```
Outputs go to `outputs/`: `customer_segments.csv`, `cluster_profile.csv`,
`recommendations_sample.csv`, `elbow_silhouette.png`, `pca_variance.png`, `pca_clusters.png`.

## Using real data
Put a CSV at `data/transactions.csv` with columns: InvoiceNo, CustomerID, InvoiceDate,
StockCode, Description, Category, Quantity, UnitPrice (the UCI "Online Retail" dataset works
after adding a Category column or removing it from `recommender.py`).

## Business actions per segment
- VIP: early access, loyalty rewards, cross-sell premium items
- Loyal: bundle offers, referral program
- Potential / Growing: personalised recommendations, free-shipping thresholds
- Low Value / Lost: win-back discounts, low-cost email campaigns
