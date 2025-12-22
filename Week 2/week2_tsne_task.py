# ===============================
# Week 2 – t-SNE Visualization on Fashion-MNIST (Colab)
# ===============================

# Step 0: Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import fashion_mnist

# ===============================
# Step 1: Load Dataset
# ===============================
(X_train, y_train), (_, _) = fashion_mnist.load_data()

X = X_train  # images
y = y_train  # labels

print("Original data shape:", X.shape)  # (60000, 28, 28)

# ===============================
# Step 2: Preprocess
# ===============================
# Flatten images
X_flat = X.reshape(len(X), -1)  # (60000, 784)

# Normalize pixel values to [0,1]
X_flat = X_flat / 255.0

print("Preprocessed data shape:", X_flat.shape)

# ===============================
# Step 3: Apply t-SNE
# ===============================
# Optional: use a subset to speed up t-SNE
subset = 5000
X_sample = X_flat[:subset]
y_sample = y[:subset]

tsne = TSNE(n_components=2, random_state=42, learning_rate='auto', init='random')
X_tsne = tsne.fit_transform(X_sample)

print("t-SNE embedding shape:", X_tsne.shape)

# ===============================
# Step 4: Visualization
# ===============================
plt.figure(figsize=(10,8))
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_sample, cmap='tab10', s=5)
plt.colorbar(scatter, ticks=range(10), label='Fashion Class')
plt.title("t-SNE Visualization of Fashion-MNIST")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()

# Reflection
# Do clusters correspond to labels?
# Some classes (e.g., sandals, sneakers) form tight clusters.
# Others overlap (e.g., shirt vs coat) because t-SNE only preserves local neighborhoods.

# Why should t-SNE not be used for training models?
# t-SNE distorts distances and scales.
# Embeddings are non-invertible, so 2D t-SNE does not reflect true high-dimensional separability.
# Using t-SNE as features can mislead classifiers.
