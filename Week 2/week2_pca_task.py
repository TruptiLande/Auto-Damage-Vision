# ===============================
# PCA ON FASHION-MNIST
# ===============================

# Step 0: Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import fashion_mnist

# ===============================
# Step 1: Load Dataset
# ===============================
(X_train, y_train), (_, _) = fashion_mnist.load_data()

X = X_train   # images
y = y_train   # labels

print("Original data shape:", X.shape)  # (60000, 28, 28)

# ===============================
# Step 2: Flatten Images
# ===============================
X_flat = X.reshape(len(X), -1)
print("Flattened data shape:", X_flat.shape)  # (60000, 784)

# Optional but recommended: normalize
X_flat = X_flat / 255.0

# ===============================
# Step 3: Apply PCA
# ===============================
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_flat)

print("PCA transformed shape:", X_pca.shape)  # (60000, 50)

# ===============================
# Step 4: Reconstruction
# ===============================
X_reconstructed = pca.inverse_transform(X_pca)

# ===============================
# Step 5: Visualization
# ===============================
n = 5  # number of images to display

plt.figure(figsize=(10, 4))

for i in range(n):
    # Original images
    plt.subplot(2, n, i + 1)
    plt.imshow(X[i], cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Reconstructed images
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Reflection

# What information is lost during PCA?
# Low-variance information such as fine textures, sharp edges, and small intensity changes is lost because PCA keeps only the most important variance directions.

# Why does reconstruction blur fine details?
# Fine details require high-frequency components, which are discarded when using a limited number of principal components. PCA preserves global shape but not local texture.
