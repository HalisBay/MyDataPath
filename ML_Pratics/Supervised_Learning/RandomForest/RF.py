import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def info_dataset(faces):
    """
    Prints basic information about the dataset.
    """
    print(f"Dataset shape: {faces.data.shape}")
    print(f"Image shape: {faces.images.shape[1:]}")
    print(f"Number of classes: {len(set(faces.target))}")

    print("\nClass distribution:")
    unique, counts = np.unique(faces.target, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} samples")


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Visualizes the confusion matrix.
    """
    labels = class_names  # Tüm sınıflar
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.title("Confusion Matrix")
    plt.show()


import seaborn as sns


def plot_overall_tp_fp_fn_tn(y_true, y_pred):
    """
    Visualizes the total TP, FP, FN, TN values for all classes as a 2x2 heatmap.
    """
    tp = np.sum(y_true == y_pred)
    fp = np.sum((y_true != y_pred) & (np.isin(y_pred, np.unique(y_true))))
    fn = np.sum((y_true != y_pred) & (np.isin(y_true, np.unique(y_pred))))
    tn = 0  # Çoklu sınıflarda TN'nin anlamı yoktur.

    metrics = np.array([[tp, tn], [fn, tn]])
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        metrics,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["FP", "FN"],
        yticklabels=["TP", "FP"],
    )
    plt.tight_layout()
    plt.show()


def compare_models(x_train, x_test, y_train, y_test):
    """
    Trains and evaluates multiple classification models on the given training and test data.
    Compares their accuracy and visualizes the results with a bar plot.
    """
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=None, random_state=42
        ),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="auto"),
    }
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {100 * acc:.2f}%")
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.ylabel("Accuracy")
    plt.title("Model Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def main():
    faces = fetch_olivetti_faces()
    # info_dataset(faces)

    x_train, x_test, y_train, y_test = train_test_split(
        faces.data, faces.target, test_size=0.2, random_state=42
    )
    print("\nModel Comparison:")
    compare_models(x_train, x_test, y_train, y_test)

    rf_clf = RandomForestClassifier(n_estimators=75, random_state=42)
    rf_clf.fit(x_train, y_train)

    y_pred = rf_clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Random Forest Accuracy With Different Parameters: {100 * acc:.2f}%")

    # plot_confusion_matrix(y_test, y_pred, class_names=np.unique(faces.target))
    plot_overall_tp_fp_fn_tn(y_test, y_pred)


if __name__ == "__main__":
    main()
