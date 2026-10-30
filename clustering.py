"""Step 3 - K-Means clustering on RFM features."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Recency", "Frequency", "Monetary"]


def prepare_features(rfm):
    """Log-transform (RFM is right-skewed) then standardise."""
    X = np.log1p(rfm[FEATURES])
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def find_best_k(X, k_range=range(2, 10), save_path="outputs/elbow_silhouette.png"):
    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, km.labels_))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(list(k_range), inertias, "o-")
    ax[0].set(title="Elbow Method", xlabel="k", ylabel="Inertia")
    ax[1].plot(list(k_range), sils, "o-", color="tab:green")
    ax[1].set(title="Silhouette Score", xlabel="k", ylabel="Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

    best_k = list(k_range)[int(np.argmax(sils))]
    return best_k, dict(zip(k_range, sils))


def fit_kmeans(X, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    return km


def profile_clusters(rfm, labels):
    out = rfm.copy()
    out["Cluster"] = labels
    prof = out.groupby("Cluster").agg(
        Customers=("Recency", "size"),
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
    ).round(1)
    prof["Share_%"] = (prof.Customers / prof.Customers.sum() * 100).round(1)
    return out, prof


def name_clusters(profile):
    """Give each cluster a business-friendly name by ranking its RFM profile."""
    names = {}
    value = profile.Monetary.rank() + profile.Frequency.rank() - profile.Recency.rank()
    order = value.sort_values(ascending=False).index.tolist()
    labels = ["VIP Customers", "Loyal Customers", "Potential / Growing", "Low Value",
              "At Risk", "Lost", "Segment 7", "Segment 8", "Segment 9"]
    for i, c in enumerate(order):
        names[c] = labels[i] if i < len(labels) else f"Segment {i + 1}"
    return names
