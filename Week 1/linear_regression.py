import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------------------------------
df = pd.read_csv("linear_data.csv")

print("\nLoaded columns:", df.columns.tolist())
print(df.head(), "\n")

# -------------------------------------------------------------------------------------
# 2. AUTO-DETECT TARGET COLUMN
# -------------------------------------------------------------------------------------
# Assumption: target is the LAST numeric column (common case)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    raise ValueError("ERROR: Not enough numeric columns found in CSV!")

target_col = numeric_cols[-1]        # Last numeric column = target
feature_cols = numeric_cols[:-1]     # All others = features

print("Using target column:", target_col)
print("Using feature columns:", feature_cols)

# Prepare data
X = df[feature_cols].values
y = df[target_col].values
idx = np.arange(len(df))

# -------------------------------------------------------------------------------------
# 3. TRAIN-TEST SPLIT
# -------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, idx, test_size=0.2, random_state=42, shuffle=True
)

# -------------------------------------------------------------------------------------
# 4. NORMALIZATION
# -------------------------------------------------------------------------------------
def normalize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1  # avoid division by zero
    return (X - mean) / std, mean, std

X_train_norm, x_mean, x_std = normalize(X_train)
X_test_norm = (X_test - x_mean) / x_std

# Add bias term
X_train_b = np.hstack([np.ones((X_train_norm.shape[0], 1)), X_train_norm])
X_test_b = np.hstack([np.ones((X_test_norm.shape[0], 1)), X_test_norm])

# -------------------------------------------------------------------------------------
# 5. LINEAR REGRESSION (Normal Equation)
# -------------------------------------------------------------------------------------
def linear_regression_closed_form(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

weights = linear_regression_closed_form(X_train_b, y_train)
print("\nLearned Weights:\n", weights)

# Predict
y_pred_train = X_train_b @ weights
y_pred_test = X_test_b @ weights

# -------------------------------------------------------------------------------------
# 6. METRICS
# -------------------------------------------------------------------------------------
mse = np.mean((y_pred_test - y_test)**2)
print(f"\nTest MSE: {mse:.4f}")

# -------------------------------------------------------------------------------------
# 7. FIXED PLOTTING
# -------------------------------------------------------------------------------------

# Sort test predictions by the original index to get clean lines
sorted_idx = np.argsort(idx_test)

plt.figure(figsize=(12, 6))
plt.plot(
    idx_test[sorted_idx],
    y_test[sorted_idx],
    "o",
    label="True Values",
    markersize=5
)
plt.plot(
    idx_test[sorted_idx],
    y_pred_test[sorted_idx],
    "x--",
    label="Predicted Values",
    markersize=6
)

plt.xlabel("Data Index")
plt.ylabel(target_col)
plt.title("True vs Predicted Values (Sorted)")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

