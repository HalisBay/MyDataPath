import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

class DecisionTreeAnalyzer:
    """
    Comprehensive Decision Tree analysis class for Iris classification
    """
    
    def __init__(self, test_size=0.3, random_state=42):
        """
        Initialize the analyzer
        
        Parameters:
        -----------
        test_size : float
            Proportion of dataset for testing
        random_state : int
            Random state for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        
    def load_and_explore_data(self):
        """Load and explore the Iris dataset"""
        print("Loading Iris dataset...")
        
        iris = load_iris()
        
        self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.df['target'] = iris.target
        self.df['species'] = iris.target_names[iris.target]
        
        # temel bilgiler
        print(f"Dataset shape: {self.df.shape}")
        print(f"Features: {list(iris.feature_names)}")
        print(f"Classes: {list(iris.target_names)}")
        
        # sınıf dağılımını kontrol ediyoruz
        print(f"\nClass distribution:")
        for i, name in enumerate(iris.target_names):
            count = sum(iris.target == i)
            print(f"  {name}: {count} samples")
        
        # Temel istatistikler
        print(f"\nDataset Statistics:")
        print(self.df.describe())
        
        # özellikler ve hedef değişkeni ayırıyoruz
        self.X = iris.data
        self.y = iris.target
        self.feature_names = iris.feature_names
        self.class_names = iris.target_names
        
        return self.df
    
    def create_data_visualizations(self):
        """Create comprehensive data exploration visualizations"""
        print("\nCreating data visualizations...")
        
        try:
            # stil ayarları
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Features dağılımları
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Iris Dataset - Feature Distributions', fontsize=16, fontweight='bold')
            
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
            
            for i, (feature, color) in enumerate(zip(self.feature_names, colors)):
                row, col = i // 2, i % 2
                axes[row, col].hist(self.df[feature], bins=20, alpha=0.7, color=color, edgecolor='black')
                axes[row, col].set_title(f'{feature} Distribution')
                axes[row, col].set_xlabel(feature)
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Sınıflara göre özellik karşılaştırması
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Feature Distributions by Species', fontsize=16, fontweight='bold')
            
            for i, feature in enumerate(self.feature_names):
                row, col = i // 2, i % 2
                
                # her sınıf için histogram çiziyoruz
                for j, species in enumerate(self.class_names):
                    species_data = self.df[self.df['species'] == species][feature]
                    axes[row, col].hist(species_data, alpha=0.6, label=species, bins=15)
                
                axes[row, col].set_title(f'{feature} by Species')
                axes[row, col].set_xlabel(feature)
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
    
    def split_data(self):
        """Split data into training and testing sets"""
        print(f"\n🔄 Splitting data (test_size={self.test_size})...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=self.y  # her sınıftan eşit oranda örnekleme
        )
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
    
    def hyperparameter_tuning(self):
        """Set optimal hyperparameters manually based on analysis"""
        print("\nSetting optimal hyperparameters...")
        
        # Manuel olarak en iyi parametreleri seç
        self.best_params = {
            'criterion': 'gini',
            'max_depth': 5,
            'max_features': 'sqrt',
            'min_samples_leaf': 1,
            'min_samples_split': 2
        }
        
        print(f"Selected parameters: {self.best_params}")
        
        # Modeli eğit
        self.best_model = DecisionTreeClassifier(**self.best_params, random_state=self.random_state)
        self.best_model.fit(self.X_train, self.y_train)
        
        return self.best_params
    
    def train_and_evaluate_model(self):
        """Evaluate the trained model performance"""
        print("\n" + "="*50)
        print("🎯 MODEL EVALUATION")
        print("="*50)
        
        # tahminleri yapıyoruz
        self.y_pred = self.best_model.predict(self.X_test)
        
        # Accuracy hesapla
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Detaylı sınıflandırma raporu
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # çapraz doğrulama skoru
        cv_scores = cross_val_score(self.best_model, self.X, self.y, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"CV Mean: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        
        return accuracy, cm
    
    def analyze_feature_importance(self):
        """Analyze and display feature importance"""
        print("\n Feature Importance Analysis...")

        # featuresleri al ve öneme göre sırala
        feature_importance = self.best_model.feature_importances_
        indices = np.argsort(feature_importance)[::-1]
        
        print("Feature Importance Ranking:")
        for i in range(len(self.feature_names)):
            print(f"{i+1}. {self.feature_names[indices[i]]}: {feature_importance[indices[i]]:.4f}")
        
        return feature_importance, indices
    
    def create_result_visualizations(self, accuracy, cm, feature_importance, indices):
        """Create comprehensive result visualizations"""
        print("\nCreating result visualizations...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Decision Tree Analysis Results', fontsize=16, fontweight='bold')
            
            # 1. Confusion Matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       ax=axes[0, 0])
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
            
            # Featuresler
            axes[0, 1].barh(range(len(feature_importance)), 
                           feature_importance[indices])
            axes[0, 1].set_yticks(range(len(feature_importance)))
            axes[0, 1].set_yticklabels([self.feature_names[i] for i in indices])
            axes[0, 1].set_xlabel('Feature Importance')
            axes[0, 1].set_title('Feature Importance')
            
            # Sınıf dağılımı (test seti)
            unique, counts = np.unique(self.y_test, return_counts=True)
            class_names_test = [self.class_names[i] for i in unique]
            axes[1, 0].pie(counts, labels=class_names_test, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Test Set Class Distribution')
            
            # Model performans özeti
            axes[1, 1].text(0.1, 0.8, f'Model: Decision Tree', fontsize=14, fontweight='bold')
            axes[1, 1].text(0.1, 0.7, f'Accuracy: {accuracy:.4f}', fontsize=12)
            axes[1, 1].text(0.1, 0.6, f'Criterion: {self.best_params["criterion"]}', fontsize=12)
            axes[1, 1].text(0.1, 0.5, f'Max Depth: {self.best_params["max_depth"]}', fontsize=12)
            axes[1, 1].text(0.1, 0.4, f'Dataset: Iris (150 samples)', fontsize=12)
            axes[1, 1].text(0.1, 0.3, f'Features: {len(self.feature_names)}', fontsize=12)
            axes[1, 1].text(0.1, 0.2, f'Classes: {len(self.class_names)}', fontsize=12)
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Model Summary')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Result visualization error: {e}")
    
    def visualize_decision_tree(self):
        """Visualize the decision tree structure"""
        print("\nVisualizing decision tree structure...")
        
        try:
            plt.figure(figsize=(20, 12))
            plot_tree(self.best_model, 
                     filled=True, 
                     feature_names=self.feature_names,
                     class_names=self.class_names,
                     rounded=True,
                     fontsize=10)
            plt.title('Decision Tree Structure', fontsize=16, fontweight='bold')
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Tree visualization error: {e}")

def main():
    """Main execution function"""
    analyzer = DecisionTreeAnalyzer()
    
    try:
        # analizkler
        df = analyzer.load_and_explore_data()
        analyzer.create_data_visualizations()
        analyzer.split_data()
        best_params = analyzer.hyperparameter_tuning()
        accuracy, cm = analyzer.train_and_evaluate_model()
        feature_importance, indices = analyzer.analyze_feature_importance()
        analyzer.create_result_visualizations(accuracy, cm, feature_importance, indices)
        analyzer.visualize_decision_tree()
        
        print("\n🎉 Analysis completed successfully!")
        print(f"Final Results: Accuracy = {accuracy:.4f}")
        print(f"🏆 Best Parameters: {best_params}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()