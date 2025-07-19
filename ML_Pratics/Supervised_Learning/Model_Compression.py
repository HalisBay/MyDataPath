import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import DecisionBoundaryDisplay


def get_datasets():
    """Generate and return a list of synthetic datasets for classification."""
    x, y = make_classification(
        n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42
    )
    x += 1.2 * np.random.uniform(size=x.shape)
    datasets = [
        (x, y),
        make_moons(noise=0.2, random_state=42),
        make_circles(noise=0.1, factor=0.2, random_state=42),
    ]
    return datasets


def get_classifiers():
    """Return a list of classifier names and their sklearn objects."""
    names = [
        "K Nearest Neighbors",
        "Linear SVM",
        "Decision Tree",
        "Random Forest",
        "Naive Bayes",
    ]
    classifiers = [
        KNeighborsClassifier(),
        SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GaussianNB(),
    ]
    return names, classifiers


def plot_decision_boundaries(datasets, names, classifiers):
    """Plot input data and decision boundaries for each classifier and dataset."""
    cm_bright = ListedColormap(["#FFA500", "#1E90FF"])
    n_datasets = len(datasets)
    n_classifiers = len(classifiers)

    fig, axes = plt.subplots(n_datasets, n_classifiers + 1, figsize=(18, 9))

    for ds_idx, (X, y) in enumerate(datasets):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Input data
        ax = axes[ds_idx, 0]
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=cm_bright,
            alpha=0.6,
            edgecolors="k",
        )
        if ds_idx == 0:
            ax.set_title("Input data")
        ax.set_xticks([])
        ax.set_yticks([])

        for clf_idx, (name, clf) in enumerate(zip(names, classifiers)):
            ax = axes[ds_idx, clf_idx + 1]
            pipe = make_pipeline(StandardScaler(), clf)
            pipe.fit(X_train, y_train)
            score = pipe.score(X_test, y_test)
            DecisionBoundaryDisplay.from_estimator(
                pipe, X, cmap=cm_bright, alpha=0.8, ax=ax, eps=0.5
            )
            ax.scatter(
                X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
            )
            ax.scatter(
                X_test[:, 0],
                X_test[:, 1],
                c=y_test,
                cmap=cm_bright,
                alpha=0.6,
                edgecolors="k",
            )
            if ds_idx == 0:
                ax.set_title(name)
            ax.text(X[:, 0].max() - 0.25, X[:, 1].min() - 0.25, f"{score:.2f}")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def main():
    datasets = get_datasets()
    names, classifiers = get_classifiers()
    plot_decision_boundaries(datasets, names, classifiers)


if __name__ == "__main__":
    main()
