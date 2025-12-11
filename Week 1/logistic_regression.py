# ===========================================================
# LOGISTIC REGRESSION (Custom Implementation)
# ===========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ===========================================================
# FEATURE MAPPING (Polynomial up to degree 4)
# ===========================================================
def feature_map(points):
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]

    features = [np.ones(len(x))]  # bias term (x^0 y^0)

    # Degree 1
    features += [x, y]

    # Degree 2
    features += [x**2, x*y, y**2]

    # Degree 3
    features += [x**3, x**2*y, x*y**2, y**3]

    # Degree 4
    features += [x**4, x**3*y, x**2*y**2, x*y**3, y**4]

    return np.column_stack(features)

# ===========================================================
# LOGISTIC REGRESSION CLASS
# ===========================================================
class LogisticRegression:

    def __init__(self) -> None:
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    # Sigmoid
    def __sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    # Probability prediction
    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        return self.__sigmoid(X @ self.weights + self.bias)

    # Class prediction
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_probability(X) >= 0.5).astype(int)

    # Loss and gradients with L2 regularization
    def __loss(self, X, y, lambda_reg=1):
        m = len(y)
        y_hat = self.predict_probability(X)

        # Loss
        loss = (-1 / m) * np.sum(
            y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9)
        )

        # Regularization term
        loss += (lambda_reg / (2 * m)) * np.sum(self.weights**2)

        # Gradients
        dw = (1 / m) * (X.T @ (y_hat - y)) + (lambda_reg / m) * self.weights
        db = (1 / m) * np.sum(y_hat - y)

        return loss, dw, db

    # Fit using gradient descent
    def fit(self, X, y, epochs=500, learning_rate=0.01,
            threshold=0.0001, lambda_reg=1):

        n_features = X.shape[1]
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0

        for _ in range(epochs):
            loss, dw, db = self.__loss(X, y, lambda_reg)

            old_weights = self.weights.copy()

            # Update
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            # Check convergence
            if np.linalg.norm(self.weights - old_weights) < threshold:
                break

# ===========================================================
# Z-SCORE NORMALIZATION
# ===========================================================
def z_score(X):
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    X_norm = (X - x_mean) / x_std
    return X_norm, x_mean, x_std

# ===========================================================
# LOAD DATA (relative path)
# ===========================================================
df = pd.read_csv("logistic_data.csv")  # CSV must be in the same folder
data = df.to_numpy()
X = data[:, :2]
y = data[:, 2]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize input data
X_train, x_mean, x_std = z_score(X_train)
X_test = (X_test - x_mean) / x_std

# Feature expansion
x_train = feature_map(X_train)
x_test = feature_map(X_test)

# ===========================================================
# DECISION BOUNDARY PLOTTING
# ===========================================================
def plot_decision_boundary(X_original, y, model, resolution=500):
    x_min, x_max = X_original[:, 0].min() - 1, X_original[:, 0].max() + 1
    y_min, y_max = X_original[:, 1].min() - 1, X_original[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    grid_original = np.c_[xx.ravel(), yy.ravel()]

    # Normalize grid using same mean and std as training
    grid_normalized = (grid_original - x_mean) / x_std

    # Feature expansion
    grid_expanded = feature_map(grid_normalized)

    Z = model.predict(grid_expanded).reshape(xx.shape)

    # Plot data points
    plt.scatter(X_original[y == 1][:, 0], X_original[y == 1][:, 1],
                c="blue", label="True", s=20)
    plt.scatter(X_original[y == 0][:, 0], X_original[y == 0][:, 1],
                c="red", label="False", s=20)

    # Plot decision boundary
    plt.contour(xx, yy, Z, levels=[0.5], colors="black")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.legend()
    plt.show()

# ===========================================================
# TRAINING THE MODEL
# ===========================================================
model = LogisticRegression()
model.fit(
    x_train, y_train,
    epochs=3000,
    learning_rate=0.1,
    threshold=1e-6,
    lambda_reg=0.8
)

# Predict and evaluate
y_pred = model.predict(x_test)
accuracy = np.mean(y_pred == y_test) * 100
print(f"Model Accuracy: {accuracy:.2f}%")

# Plot final decision boundary
plot_decision_boundary(X, y, model)
