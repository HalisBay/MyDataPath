import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.cluster import (
    MiniBatchKMeans,
    SpectralClustering,
    AgglomerativeClustering,
    DBSCAN,
    Birch,
)
from scipy.cluster.hierarchy import ward, fcluster
import warnings

warnings.filterwarnings("ignore")


class DatasetGenerator:
    """The class that creates different synthetic datasets."""

    def __init__(self, n_samples=1500, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state

    def generate(self):
        X_circles, _ = make_circles(
            n_samples=self.n_samples,
            factor=0.5,
            noise=0.05,
            random_state=self.random_state,
        )
        X_moons, _ = make_moons(
            n_samples=self.n_samples, noise=0.05, random_state=self.random_state
        )
        X_blobs, _ = make_blobs(
            n_samples=self.n_samples, centers=3, random_state=self.random_state
        )
        X_random = np.random.rand(self.n_samples, 2)

        return [
            (X_circles, "Noisy Circles"),
            (X_moons, "Noisy Moons"),
            (X_blobs, "Blobs"),
            (X_random, "Random 2D"),
        ]


class ClusteringAlgorithms:
    """The class that manages clustring algorithms and parameters."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def get_algorithms(self):
        # Farklı algoritmalar, bazıları parametre ayarı gerektiriyor (ör: n_clusters)
        return [
            (
                "MiniBatchKMeans",
                MiniBatchKMeans(n_clusters=3, random_state=self.random_state),
            ),
            (
                "SpectralClustering",
                SpectralClustering(
                    n_clusters=3,
                    affinity="nearest_neighbors",
                    assign_labels="kmeans",
                    random_state=self.random_state,
                ),
            ),
            ("Ward", None),  # Ward için sklearn yerine scipy kullanılıyor
            ("Agglomerative", AgglomerativeClustering(n_clusters=3)),
            ("DBSCAN", DBSCAN(eps=0.3)),
            ("Birch", Birch(n_clusters=3)),
        ]


class ClusterPlotter:
    """The class that visualizes clustering results."""

    def __init__(self, datasets, algorithms):
        self.datasets = datasets
        self.algorithms = algorithms

    def plot(self):
        fig, axes = plt.subplots(
            len(self.datasets), len(self.algorithms), figsize=(18, 12)
        )
        plt.subplots_adjust(
            left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.15
        )

        for row, (X, name) in enumerate(self.datasets):
            for col, (algo_name, algorithm) in enumerate(self.algorithms):
                ax = axes[row, col]
                if algo_name == "Ward":
                    # Sklearn'de Ward algoritması yok bu yüzden scipy ile uygulanıyor
                    linkage_matrix = ward(X)
                    labels = fcluster(linkage_matrix, t=3, criterion="maxclust")
                else:
                    # fit_predict varsa doğrudan kullan, yoksa fit sonrası labels_ al
                    if hasattr(algorithm, "fit_predict"):
                        labels = algorithm.fit_predict(X)
                    else:
                        labels = algorithm.fit(X).labels_
                ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=10)
                if row == 0:
                    ax.set_title(algo_name, fontsize=10)
                if col == 0:
                    ax.set_ylabel(name, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        plt.show()


def main():
    # Veri setlerini oluştur
    dataset_gen = DatasetGenerator()
    datasets = dataset_gen.generate()

    # Algoritmaları hazırla
    algo_manager = ClusteringAlgorithms()
    algorithms = algo_manager.get_algorithms()

    # Sonuçları çiz
    plotter = ClusterPlotter(datasets, algorithms)
    plotter.plot()


if __name__ == "__main__":
    main()
