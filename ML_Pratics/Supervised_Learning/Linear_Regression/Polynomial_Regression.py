import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

def generate_data(seed=42, n_samples=100):
    np.random.seed(seed)
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y_true = 2 * X**3 - 4 * X**2 + X + 5
    y = y_true + np.random.normal(scale=8, size=X.shape)
    return X, y, y_true

def polynomial_grid_search(X_train, y_train, max_degree=8):
    degrees = np.arange(1, max_degree + 1)
    pipe = make_pipeline(PolynomialFeatures(), LinearRegression())
    param_grid = {'polynomialfeatures__degree': degrees}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train.ravel())
    return grid

def plot_cv_scores(grid, max_degree=8):
    means = -grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    degrees = np.arange(1, max_degree + 1)
    plt.figure(figsize=(8, 4))
    plt.errorbar(degrees, means, yerr=stds, fmt='-o', capsize=5)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('CV Mean Squared Error')
    plt.title('Cross-Validation Error vs. Polynomial Degree')
    plt.xticks(degrees)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_results(X, y, y_true, y_fit, best_degree):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='gray', alpha=0.5, label='Noisy data')
    plt.plot(X, y_true, color='green', linewidth=2, label='True function')
    plt.plot(X, y_fit, color='red', linewidth=2, label=f'Polynomial fit (degree={best_degree})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Polynomial Regression Fit (Degree={best_degree})')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Data üretimi
    X, y, y_true = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Model seçimi
    grid = polynomial_grid_search(X_train, y_train, max_degree=8)
    best_degree = grid.best_params_['polynomialfeatures__degree']
    print(f"Best polynomial degree (CV): {best_degree}")

    # skorlar
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_fit = best_model.predict(X)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.2f}")
    print(f"Test R2 Score: {r2:.2f}")

    # Görselleştr
    plot_cv_scores(grid, max_degree=8)
    plot_results(X, y, y_true, y_fit, best_degree)

if __name__ == "__main__":
    main()