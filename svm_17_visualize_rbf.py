"""
svm_17_visualize_rbf.py
-----------------------
Concept: Non-Linear RBF Boundary Visualization.
Algorithm: Support Vector Classifier (SVC) with an RBF Kernel.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates a complex concentric circle layout.
2. Fits an RBF SVM.
3. Obtains predictions and distance scores over a dense coordinate grid.
4. Colors the background zones.
5. Draws the boundary contours.
6. Saves the resulting visual artifact.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


print("--- Step 1: Generating Concentric Circles ---")
X, y = make_circles(n_samples=150, noise=0.1, factor=0.4, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data generated and standardized.")

print("\n--- Step 2: Training RBF SVM ---")
model = SVC(kernel='rbf', C=1.5, gamma=1.0)
model.fit(X_scaled, y)
print("RBF model trained successfully.")

print("\n--- Step 3: Shading Non-Linear Decision Regions ---")
plt.figure(figsize=(10, 8))

# Create grid points for background shading
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# decision_function gives distance/signed score
Z = model.decision_function(grid_points).reshape(xx.shape)

# Color the regions based on predicted side
plt.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#e74c3c', '#3498db'], alpha=0.15)

# Draw boundary curve where decision score = 0
plt.contour(xx, yy, Z, levels=[0], colors=['black'], linewidths=[2.5])

# Scatter plot the data points
plt.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], color='#e74c3c', 
            edgecolors='k', s=60, label='Class 0')
plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], color='#3498db', 
            edgecolors='k', s=60, label='Class 1')

# Highlight support vectors
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=150,
           linewidth=1.5, facecolors='none', edgecolors='black', label='Support Vectors')

plt.title('RBF SVM Non-Linear Decision Boundary & Support Vectors', fontsize=14, fontweight='bold')
plt.xlabel('Scaled Feature 1')
plt.ylabel('Scaled Feature 2')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

