"""
svm_12_rbf_kernel_xor.py
------------------------
Concept: RBF Kernel on the XOR Problem.
Algorithm: Support Vector Classifier (SVC) with an RBF Kernel.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates a synthetic dataset following the XOR logic gate distribution (with random noise).
2. Standardizes coordinates.
3. Fits an RBF SVM to solve the non-linear XOR separation constraint.
4. Checks classification accuracy.
5. Displays a contour plot depicting the diagonal boundary zones.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


print("--- Step 1: Simulating XOR Coordinate Dataset ---")
np.random.seed(42)

# Generate 200 data points in 4 quadrants
X = np.random.randn(200, 2)
# Label: True (1) if X1 and X2 have opposite signs, else False (0)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)

df = pd.DataFrame(X, columns=['Feature_X1', 'Feature_X2'])
df['Label'] = y

print("Sample XOR Coordinates:")
print(df.head())
print(f"\nLabel Counts:\n{df['Label'].value_counts()}")

print("\n--- Step 2: Training RBF SVM on XOR Data ---")
# C controls boundary hardness, gamma controls locality (small means large boundary range)
model = SVC(kernel='rbf', C=10.0, gamma=1.0)
model.fit(X, y)
print("RBF model training complete.")

print("\n--- Step 3: Model Evaluation ---")
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Overall Training Accuracy: {accuracy * 100:.1f}%")

print("\n--- Step 4: Visualizing Diagonal Boundaries ---")
plt.figure(figsize=(8, 8))

# Create grid coordinates
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = model.decision_function(grid_points).reshape(xx.shape)

# Plot decision boundaries and margin surfaces
plt.contourf(xx, yy, Z > 0, alpha=0.15, cmap='coolwarm')
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

# Plot coordinate points
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='#e74c3c', label='Class 0 (False)', edgecolors='k', s=50)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='#3498db', label='Class 1 (True)', edgecolors='k', s=50)

plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)

plt.title('Solving the XOR Problem with RBF SVM', fontsize=14, fontweight='bold')
plt.xlabel('Feature X1')
plt.ylabel('Feature X2')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

