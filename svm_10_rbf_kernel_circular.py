"""
svm_10_rbf_kernel_circular.py
-----------------------------
Concept: RBF Kernel on Circular Data.
Algorithm: Support Vector Classifier (SVC) with RBF vs. Linear Kernels.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates concentric circular datasets using `make_circles`.
2. Standardizes features to maintain spatial uniformity.
3. Fits both a Linear SVM and an RBF SVM to classify the circles.
4. Compares and prints the accuracy of both approaches.
5. Displays a side-by-side plot comparing decision boundaries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


print("--- Step 1: Generating Concentric Circles ---")
# Generate circular binary dataset
X, y = make_circles(n_samples=200, noise=0.08, factor=0.5, random_state=42)

df = pd.DataFrame(X, columns=['X1', 'X2'])
df['Label'] = y
print("Data sample:")
print(df.head())

print("\n--- Step 2: Preparing and Scaling Data ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data standardized.")

print("\n--- Step 3: Fitting Linear SVM (Control Group) ---")
linear_model = SVC(kernel='linear', C=1.0)
linear_model.fit(X_train_scaled, y_train)
linear_acc = accuracy_score(y_test, linear_model.predict(X_test_scaled))
print(f"Linear SVM Accuracy: {linear_acc * 100:.1f}% (Expected to perform poorly)")

print("\n--- Step 4: Fitting RBF SVM (Experimental Group) ---")
# The RBF (radial basis function) kernel projects data into infinite dimensions where it is separable.
rbf_model = SVC(kernel='rbf', C=1.0, gamma='scale')
rbf_model.fit(X_train_scaled, y_train)
rbf_acc = accuracy_score(y_test, rbf_model.predict(X_test_scaled))
print(f"RBF SVM Accuracy: {rbf_acc * 100:.1f}% (Expected to perform very well)")

print("\n--- Step 5: Visualizing and Saving Comparison Plot ---")
# We will create a mesh grid to draw decision boundaries for both models
xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 100), np.linspace(-2.5, 2.5, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Get decisions
Z_linear = linear_model.decision_function(grid_points).reshape(xx.shape)
Z_rbf = rbf_model.decision_function(grid_points).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Linear SVM
axes[0].scatter(X_test_scaled[y_test == 0, 0], X_test_scaled[y_test == 0, 1], color='#e74c3c', label='Class 0')
axes[0].scatter(X_test_scaled[y_test == 1, 0], X_test_scaled[y_test == 1, 1], color='#3498db', label='Class 1')
axes[0].contour(xx, yy, Z_linear, levels=[0], colors='black', linewidths=2)
axes[0].set_title(f'Linear Kernel Boundary\nAccuracy: {linear_acc * 100:.1f}%', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Scaled X1')
axes[0].set_ylabel('Scaled X2')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)

# Subplot 2: RBF SVM
axes[1].scatter(X_test_scaled[y_test == 0, 0], X_test_scaled[y_test == 0, 1], color='#e74c3c', label='Class 0')
axes[1].scatter(X_test_scaled[y_test == 1, 0], X_test_scaled[y_test == 1, 1], color='#3498db', label='Class 1')
axes[1].contour(xx, yy, Z_rbf, levels=[0], colors='black', linewidths=2)
axes[1].set_title(f'RBF Kernel Boundary\nAccuracy: {rbf_acc * 100:.1f}%', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Scaled X1')
axes[1].set_ylabel('Scaled X2')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Comparison of Linear vs. RBF SVM on Circular Data', fontsize=14, fontweight='bold')
plt.tight_layout()

plt.show()
print("Program completed successfully!")

