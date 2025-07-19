import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

def create_decision_boundary_visualization():
    """
    Visualizes Decision Tree decision boundaries for the Iris dataset feature pairs.
    Returns accuracy scores and feature pairs.
    """
    
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {target_names}")
    print(f"Feature pairs to analyze: {len(list(combinations(range(4), 2)))}")
    
    # Sınıf sayısı ve renkler
    num_classes = len(target_names)
    colors = ["red", "gold", "blue"]  # Daha net renkler
    
    # Tüm ikili özellik kombinasyonlarını al (4 özellik => 6 çift)
    feature_pairs = list(combinations(range(4), 2))
    
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 14))
    
    accuracies = []  # Accuracy skorlarını kaydet
    
    for pairidx, (i, j) in enumerate(feature_pairs):
        # İki özelliği seç
        X_pair = X[:, [i, j]]
        
        # Veriyi eğitim-test olarak böl
        X_train, X_test, y_train, y_test = train_test_split(
            X_pair, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Sınıflandırıcıyı eğit
        clf = DecisionTreeClassifier(random_state=42, max_depth=3).fit(X_train, y_train)
        
        # Test accuracy hesapla
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        
        ax = plt.subplot(2, 3, pairidx + 1)
        
        # Karar sınırı çizimi
        DecisionBoundaryDisplay.from_estimator(
            clf,
            X_pair,
            cmap=plt.cm.RdYlBu,
            response_method="predict",
            ax=ax,
            xlabel=feature_names[i],
            ylabel=feature_names[j],
            alpha=0.8
        )
        
        # Veri noktalarının çizimi
        for class_idx, (color, class_name) in enumerate(zip(colors, target_names)):
            class_mask = (y == class_idx)
            ax.scatter(
                X_pair[class_mask, 0],
                X_pair[class_mask, 1],
                c=color,
                label=f"{class_name}",
                edgecolors="black",
                s=50,
                alpha=0.9
            )
        
        ax.set_title(f"{feature_names[i]} vs {feature_names[j]}\nAccuracy: {accuracy:.3f}", 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Decision Tree - Iris Feature Pairs Decision Boundaries", 
                fontsize=16, fontweight='bold', y=0.97)
    plt.tight_layout(h_pad=5.0, w_pad=2.5, pad=7.0)
    
    avg_accuracy = np.mean(accuracies)
    print(f"\nResults:")
    print(f"Average Test Accuracy: {avg_accuracy:.3f}")
    print(f"Highest Accuracy: {max(accuracies):.3f}")
    print(f"Lowest Accuracy: {min(accuracies):.3f}")
    
    # En iyi özellik çiftini buluyoruz
    best_pair_idx = np.argmax(accuracies)
    best_i, best_j = feature_pairs[best_pair_idx]
    print(f"Best feature pair: {feature_names[best_i]} + {feature_names[best_j]} (Acc: {max(accuracies):.3f})")
    
    plt.show()
    
    return accuracies, feature_pairs

def analyze_feature_importance():
    """
    Analyzes feature importances using Decision Tree trained on all features of the Iris dataset.
    Returns importances and sorted indices.
    """
    print(f"\nFeature Importance Analysis")
    print("-" * 30)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Model hiperparametre seç ve eğit
    clf = DecisionTreeClassifier(random_state=42, max_depth=3)
    clf.fit(X, y)
    
    # Özellik önemlerini al ve sırala (en önemliden en önemsize )
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Feature Importance Ranking:")
    for i in range(len(iris.feature_names)):
        print(f"{i+1}. {iris.feature_names[indices[i]]}: {importances[indices[i]]:.3f}")
    
    return importances, indices

if __name__ == "__main__":
    # Genel analiz
    accuracies, pairs = create_decision_boundary_visualization()
    
    # Feature analizi
    importances, indices = analyze_feature_importance()
    
    print(f"\nAnalysis completed")
