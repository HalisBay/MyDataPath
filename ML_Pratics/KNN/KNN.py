import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class KNNAnalyzer:
    """
    A comprehensive KNN analysis class for breast cancer classification
    """

    def __init__(self, test_size=0.3, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.best_k = None
        self.best_model = None

    def load_and_prepare_data(self):
        """Load and prepare the breast cancer dataset"""
        print("Loading breast cancer dataset...")
        cancer = load_breast_cancer()

        # veri çerçevesi oluşturuyoruz, veri daha anlaşılır olsun diye
        self.df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
        self.df["target"] = cancer.target

        print(f"Dataset shape: {self.df.shape}")
        print(f"Features: {len(cancer.feature_names)}")
        print(f"Classes: {cancer.target_names}")
        print(f"Class distribution:\n{pd.Series(cancer.target).value_counts()}")

        # özellikler ve hedef değişkeni hazırlıyoruz
        self.X = cancer.data
        self.y = cancer.target

    def split_and_scale_data(self):
        """Split data into train/test sets and apply standardization"""
        print("\nSplitting and scaling data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

        # standartlaştırma işlemi yapıyoruz
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"Training set size: {self.X_train_scaled.shape[0]}")
        print(f"Test set size: {self.X_test_scaled.shape[0]}")

    def find_optimal_k(self, k_range=(1, 31)):
        """Find optimal K value using cross-validation"""
        print(f"\nFinding optimal K value in range {k_range}...")

        # k değerlerini ve skorları tutuyoruz
        k_values = range(k_range[0], k_range[1])
        cv_scores = []
        test_scores = []

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)

            # çapraz doğrulama skoru alıyoruz
            cv_score = cross_val_score(knn, self.X_train_scaled, self.y_train, cv=5)
            cv_scores.append(cv_score.mean())

            # test skoru hesaplıyoruz
            knn.fit(self.X_train_scaled, self.y_train)
            test_pred = knn.predict(self.X_test_scaled)
            test_scores.append(accuracy_score(self.y_test, test_pred))

        # en iyi k değerini buluyoruz
        best_k_idx = np.argmax(cv_scores)
        self.best_k = k_values[best_k_idx]

        print(f"Best K value: {self.best_k}")
        print(f"Best CV score: {cv_scores[best_k_idx]:.4f}")

        return k_values, cv_scores, test_scores

    def train_best_model(self):
        """Train the model with optimal K value"""
        print(f"\nTraining final model with K={self.best_k}...")
        self.best_model = KNeighborsClassifier(n_neighbors=self.best_k)
        self.best_model.fit(self.X_train_scaled, self.y_train)

        # tahminleri alıyoruz
        self.y_pred = self.best_model.predict(self.X_test_scaled)
        self.y_pred_proba = self.best_model.predict_proba(self.X_test_scaled)[:, 1]

    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\n" + "=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)

        # temel metrikleri hesaplıyoruz
        accuracy = accuracy_score(self.y_test, self.y_pred)
        auc_score = roc_auc_score(self.y_test, self.y_pred_proba)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")

        # sınıflandırma raporu
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred))

        # karışıklık matrisi
        cm = confusion_matrix(self.y_test, self.y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)

        return accuracy, auc_score, cm

    def create_visualizations(self, k_values, cv_scores, test_scores):
        """Create comprehensive visualizations"""
        # stil ayarlıyoruz
        plt.style.use("default")
        sns.set_palette("husl")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "KNN Analysis - Breast Cancer Classification",
            fontsize=16,
            fontweight="bold",
        )

        # k değeri optimizasyon grafiği
        axes[0, 0].plot(
            k_values,
            cv_scores,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=6,
            label="CV Score",
        )
        axes[0, 0].plot(
            k_values,
            test_scores,
            marker="s",
            linestyle="--",
            linewidth=2,
            markersize=6,
            label="Test Score",
        )
        axes[0, 0].axvline(
            x=self.best_k, color="red", linestyle=":", label=f"Best K={self.best_k}"
        )
        axes[0, 0].set_xlabel("K Value")
        axes[0, 0].set_ylabel("Accuracy Score")
        axes[0, 0].set_title("K-Value Optimization")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # karışıklık matrisi
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 1])
        axes[0, 1].set_title("Confusion Matrix")
        axes[0, 1].set_xlabel("Predicted")
        axes[0, 1].set_ylabel("Actual")

        # roc eğrisi
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        auc_score = roc_auc_score(self.y_test, self.y_pred_proba)
        axes[1, 0].plot(
            fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc_score:.3f})"
        )
        axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
        axes[1, 0].set_xlabel("False Positive Rate")
        axes[1, 0].set_ylabel("True Positive Rate")
        axes[1, 0].set_title("ROC Curve")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # en önemli 10 özelliği gösteriyoruz (hedef ile korelasyon)
        feature_corr = self.df.corr()["target"].abs().sort_values(ascending=False)[1:11]
        axes[1, 1].barh(range(len(feature_corr)), feature_corr.values)
        axes[1, 1].set_yticks(range(len(feature_corr)))
        axes[1, 1].set_yticklabels(
            [
                name[:20] + "..." if len(name) > 20 else name
                for name in feature_corr.index
            ]
        )
        axes[1, 1].set_xlabel("Correlation with Target")
        axes[1, 1].set_title("Top 10 Features by Correlation")

        plt.tight_layout()
        plt.show()


def main():
    """Main execution function"""

    analyzer = KNNAnalyzer()

    # analiz adımlarını sırayla çalıştırıyoruz
    analyzer.load_and_prepare_data()
    analyzer.split_and_scale_data()
    k_values, cv_scores, test_scores = analyzer.find_optimal_k()
    analyzer.train_best_model()
    accuracy, auc_score, cm = analyzer.evaluate_model()
    analyzer.create_visualizations(k_values, cv_scores, test_scores)

    print("\nAnalysis completed successfully\n")
    print(f"Results: Accuracy = {accuracy:.4f}, AUC = {auc_score:.4f}")


if __name__ == "__main__":
    main()
