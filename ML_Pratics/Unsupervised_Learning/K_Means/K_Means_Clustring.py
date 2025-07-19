import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y_true = make_blobs(n_samples=300, centers=5, cluster_std=0.7, random_state=42)

kmeans = KMeans(n_clusters=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before clustering
axes[0].scatter(X[:, 0], X[:, 1], c='gray', s=40, edgecolor='k', alpha=0.7)
axes[0].set_title("Before Clustering")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")

# After clustering
scatter = axes[1].scatter(X[:, 0], X[:, 1], c=y_kmeans, s=40, edgecolor='k', alpha=0.7)
axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', s=200, marker='X', label='Centers')
axes[1].set_title("After K-Means Clustering")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].legend()

plt.show()