"""
svm_16_visualize_linear.py
--------------------------
Concept: Detailed 2D Boundary Visualization.
Algorithm: Support Vector Classifier (SVC) with a Linear Kernel.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates 2D blobs.
2. Fits a linear SVM model.
3. Computes a dense meshgrid over the coordinate space.
4. Colors the background regions (filled contours) according to model predictions.
5. Draws the exact decision hyperplane and margins.
6. Highlights the support vectors.
7. Displays a premium high-resolution visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


print("--- Step 1: Generating 2D Dataset ---")
X, y = make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=1.0, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data generated and standardized.")

print("\n--- Step 2: Fitting Linear SVM ---")
model = SVC(kernel='linear', C=1.0)
model.fit(X_scaled, y)
print("Linear SVM trained.")

print("\n--- Step 3: Plotting and Shading Decision Boundary ---")
plt.figure(figsize=(10, 8))

# Create grid points for background shading
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# decision_function gives distance/signed score to hyperplane
Z = model.decision_function(grid_points).reshape(xx.shape)

# Color the regions based on predicted side
plt.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#e74c3c', '#3498db'], alpha=0.15)

# Plot decision boundary (Z=0) and margin bounds (Z=-1, Z=1)
plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['black', 'black', 'black'], 
            linestyles=['--', '-', '--'], linewidths=[1.5, 2.5, 1.5])

# Scatter plot the data points
plt.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], color='#e74c3c', 
            edgecolors='k', s=60, label='Class 0')
plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], color='#3498db', 
            edgecolors='k', s=60, label='Class 1')

# Draw support vectors
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=200,
           linewidth=2, facecolors='none', edgecolors='black', label='Support Vectors')

plt.title('Premium Linear SVM Decision Boundary, Margins & Support Vectors', fontsize=14, fontweight='bold')
plt.xlabel('Scaled Feature 1')
plt.ylabel('Scaled Feature 2')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

