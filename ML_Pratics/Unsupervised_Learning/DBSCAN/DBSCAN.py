import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN

def main():
    X, y_true = make_circles(n_samples=1000, factor=0.5, noise=0.08, random_state=42)

    dbscan = DBSCAN(eps=0.1, min_samples=5)
    clusters = dbscan.fit_predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Before
    axes[0].scatter(X[:, 0], X[:, 1],cmap="tab20")
    axes[0].set_title("Before (True Labels)")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")

    # After
    axes[1].scatter(X[:, 0], X[:, 1], c=clusters, cmap="tab20")
    axes[1].set_title("After (DBSCAN Clusters)")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()