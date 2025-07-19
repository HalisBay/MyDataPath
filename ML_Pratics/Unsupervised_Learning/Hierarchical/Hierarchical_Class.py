import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


def plot_dendrogram(Z, ax, method, n_clusters):
    dendrogram(
        Z,
        ax=ax,
        color_threshold=None,
        truncate_mode="lastp",
        p=40,
        show_leaf_counts=False,
    )
    ax.set_title(f"Dendrogram ({method})")
    ax.set_xlabel("Sample Index or (Cluster Size)")
    ax.set_ylabel("Distance")
    ax.set_xticks([])  # X ekseni etiketlerini gizler

def main():
    X, y_true = make_blobs(n_samples=300, centers=5, cluster_std=0.6, random_state=42)
    X = StandardScaler().fit_transform(X)

    linkage_methods = ["ward", "single", "complete", "average"]
    n_methods = len(linkage_methods)
    n_clusters = 7

    fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 10))

    for i, method in enumerate(linkage_methods):
        # Dendrogram
        Z = linkage(X, method=method)
        plot_dendrogram(Z, axes[0, i], method, n_clusters)

        # Clustering
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
        clusters = model.fit_predict(X)

        axes[1, i].scatter(
            X[:, 0], X[:, 1], c=clusters, cmap="tab10", s=40, edgecolor="k", alpha=0.7
        )
        axes[1, i].set_title(f"Clusters ({method})")
        axes[1, i].set_xlabel("Feature 1")
        axes[1, i].set_ylabel("Feature 2")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
