from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def load_data():
    """Fetch and prepare the Heart Disease dataset."""
    heart_disease = fetch_ucirepo(id=45)
    df = pd.DataFrame(data=heart_disease.data.features)
    df["target"] = heart_disease.data.targets
    if df.isna().any().any():
        df.dropna(inplace=True)
        print("Dropped missing values.")
    return df


def tune_hyperparameters(X_train, y_train):
    """Find best hyperparameters using GridSearchCV."""
    param_grid = {
        "penalty": ["l1", "l2"],
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"],
        "max_iter": [100, 200],
    }
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)
    print("Best Params:", grid.best_params_)
    return grid.best_estimator_


def evaluate_model(model, X_test, y_test):
    """Evaluate model with multiple metrics and visualizations."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {100 * acc:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap=plt.cm.Blues)

    plt.title("Confusion Matrix")
    plt.show()


def main():
    print("Heart Disease Prediction with Logistic Regression")
    df = load_data()
    X = df.drop("target", axis=1).values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    model = tune_hyperparameters(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
