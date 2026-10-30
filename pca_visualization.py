"""Step 4 - PCA for dimensionality reduction & cluster visualisation."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def run_pca(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    comps = pca.fit_transform(X)
    return pca, comps


def plot_explained_variance(X, save_path="outputs/pca_variance.png"):
    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(5, 4))
    plt.bar(range(1, len(cum) + 1), pca.explained_variance_ratio_, alpha=0.6, label="Individual")
    plt.step(range(1, len(cum) + 1), cum, where="mid", color="red", label="Cumulative")
    plt.xlabel("Principal component"); plt.ylabel("Explained variance")
    plt.title("PCA Explained Variance"); plt.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()


def plot_clusters(comps, labels, names, pca, save_path="outputs/pca_clusters.png"):
    df = pd.DataFrame(comps, columns=["PC1", "PC2"])
    df["Segment"] = [names[l] for l in labels]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="Segment", palette="Set2", s=25, alpha=0.8)
    ev = pca.explained_variance_ratio_ * 100
    plt.xlabel(f"PC1 ({ev[0]:.1f}%)"); plt.ylabel(f"PC2 ({ev[1]:.1f}%)")
    plt.title("Customer Segments (PCA projection)")
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()


def pca_loadings(pca, feature_names):
    return pd.DataFrame(pca.components_.T, index=feature_names,
                        columns=[f"PC{i + 1}" for i in range(pca.n_components_)]).round(3)
