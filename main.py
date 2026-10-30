"""Run the full pipeline:  python main.py"""
import os
import pandas as pd
from data_generator import generate_transactions
from rfm_analysis import clean_transactions, rfm_pipeline
from clustering import (prepare_features, find_best_k, fit_kmeans,
                        profile_clusters, name_clusters, FEATURES)
from pca_visualization import run_pca, plot_explained_variance, plot_clusters, pca_loadings
from recommender import Recommender

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 1. Load data (generate if missing)
path = "data/transactions.csv"
if not os.path.exists(path):
    generate_transactions().to_csv(path, index=False)
df = clean_transactions(pd.read_csv(path))
print(f"Transactions: {len(df):,} | Customers: {df.CustomerID.nunique():,}")

# 2. RFM
rfm = rfm_pipeline(df)
print("\n=== RFM rule-based segments ===")
print(rfm.RFM_Segment.value_counts())

# 3. K-Means
X, scaler = prepare_features(rfm)
best_k, sils = find_best_k(X)
print(f"\nSilhouette by k: { {k: round(v, 3) for k, v in sils.items()} }")
K = 4  # business-friendly choice; change to best_k if you prefer the statistical optimum
print(f"Best k by silhouette = {best_k}; using k = {K}")
km = fit_kmeans(X, K)
rfm_clustered, profile = profile_clusters(rfm, km.labels_)
names = name_clusters(profile)
profile["Segment"] = profile.index.map(names)
print("\n=== Cluster profile ===")
print(profile)
rfm_clustered["Segment"] = rfm_clustered.Cluster.map(names)

# 4. PCA
plot_explained_variance(X)
pca, comps = run_pca(X, 2)
plot_clusters(comps, km.labels_, names, pca)
print("\n=== PCA loadings ===")
print(pca_loadings(pca, FEATURES))
print(f"Variance explained by 2 PCs: {pca.explained_variance_ratio_.sum():.1%}")

# 5. Recommendation
rec = Recommender(df, rfm_clustered["Segment"])
print(f"\nHit-rate@10 (hold-out): {rec.evaluate_hit_rate():.1%}")
sample_cust = rfm_clustered[rfm_clustered.Segment == "VIP Customers"].index[0]
print(f"\nTop-5 recommendations for customer {sample_cust} ({rfm_clustered.loc[sample_cust,'Segment']}):")
print(rec.recommend(sample_cust, 5))

# 6. Save outputs
rfm_clustered.to_csv("outputs/customer_segments.csv")
profile.to_csv("outputs/cluster_profile.csv")
all_recs = {c: rec.recommend(c, 5).index.tolist() for c in rfm_clustered.index[:200]}
pd.DataFrame.from_dict(all_recs, orient="index",
                       columns=[f"Rec_{i}" for i in range(1, 6)]).to_csv("outputs/recommendations_sample.csv")
print("\nSaved results to outputs/")
