import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px


def load_and_preprocess(filepath):
    data = pd.read_csv(filepath)
    if data.isnull().sum().sum() > 0:
        data = data.fillna(data.mean())
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return data


def scale_features(data):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.values)
    return X_scaled


def reduce_dimensions(X_scaled):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    return X_pca, X_tsne


def kmeans_analysis(X_scaled, K=range(2, 11)):
    inertia = []  # inertia: Küme içi toplam hata (düşük olması iyi)
    # sihoutte : ait oldukları kümeye ne kadar iyi uyduğunu ve diğer kümelerden ne kadar ayrıldığını ölçer
    silhouette = []
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        inertia.append(kmeans.inertia_)
        silhouette.append(silhouette_score(X_scaled, labels))
    return inertia, silhouette


def plot_kmeans_metrics(K, inertia, silhouette):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K, inertia, "bo-")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.subplot(1, 2, 2)
    plt.plot(K, silhouette, "ro-")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score")
    plt.tight_layout()
    plt.show()


def fit_kmeans(X_scaled, data, optimal_k=5):
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    data["KMeans_Cluster"] = labels
    return data


def fit_dbscan(X_scaled, data, eps=1.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    db_labels = dbscan.fit_predict(X_scaled)
    data["DBSCAN_Cluster"] = db_labels
    return data


def plot_clusters(X_proj, labels, title):
    fig = px.scatter(
        x=X_proj[:, 0],
        y=X_proj[:, 1],
        color=labels.astype(str),
        title=title,
        labels={"color": "Cluster"},
    )
    fig.show()


def cluster_profiling(data):
    profile = data.groupby("KMeans_Cluster").agg(["mean", "std", "count"])
    print(profile)
    print(data["KMeans_Cluster"].value_counts())


def plot_dendrogram(X_scaled):
    plt.figure(figsize=(10, 5))
    linkage_matrix = linkage(X_scaled, method="ward")
    dendrogram(linkage_matrix)
    plt.title("Dendrogram - Customer Segmentation")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.show()


def main():
    data = load_and_preprocess("customer_segments.csv")
    X_scaled = scale_features(data)
    X_pca, X_tsne = reduce_dimensions(X_scaled)

    # KMeans analysis
    K = range(2, 11)
    inertia, silhouette = kmeans_analysis(X_scaled, K)
    plot_kmeans_metrics(K, inertia, silhouette)

    # Fit KMeans and DBSCAN
    data = fit_kmeans(X_scaled, data, optimal_k=5)
    data = fit_dbscan(X_scaled, data, eps=1.5, min_samples=5)

    # Visualize KMeans clusters
    plot_clusters(X_pca, data["KMeans_Cluster"], "KMeans Clusters (PCA Projection)")
    plot_clusters(X_tsne, data["KMeans_Cluster"], "KMeans Clusters (t-SNE Projection)")

    # Visualize DBSCAN clusters
    plot_clusters(X_pca, data["DBSCAN_Cluster"], "DBSCAN Clusters (PCA Projection)")
    plot_clusters(X_tsne, data["DBSCAN_Cluster"], "DBSCAN Clusters (t-SNE Projection)")

    # Profiling and dendrogram
    cluster_profiling(data)
    plot_dendrogram(X_scaled)


if __name__ == "__main__":
    main()
