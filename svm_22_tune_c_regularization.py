"""
svm_22_tune_c_regularization.py
--------------------------------
Concept: Tuning the Regularization Parameter (C).
Algorithm: Support Vector Classifier (SVC) with a Linear Kernel.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates a slightly overlapping synthetic 2D dataset.
2. Standardizes coordinates.
3. Fits three separate linear SVMs with different C values: C=0.01 (Soft Margin), C=1.0 (Balanced), C=100.0 (Hard Margin).
4. Compares and prints the number of support vectors and test accuracy for each.
5. Displays a 3-panel comparison plot displaying the boundaries, margin widths, and support vectors.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


print("--- Step 1: Generating Slightly Overlapping Data ---")
X, y = make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=1.4, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data scaled successfully.")

print("\n--- Step 2: Training SVMs with Different C Values ---")
c_values = [0.01, 1.0, 100.0]
models = {}

for c in c_values:
    # Use a linear kernel to see the geometric margin shift clearly
    model = SVC(kernel='linear', C=c)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    num_sv = len(model.support_vectors_)

    models[c] = {
        'model_obj': model,
        'accuracy': acc,
        'support_vectors_count': num_sv
    }

    print(f"SVM with C = {c:<6} | Test Accuracy: {acc*100:.1f}% | Support Vectors Count: {num_sv}")
    # Note: Smaller C allows more margin violations, resulting in more support vectors (Soft Margin).
    # Larger C penalizes violations heavily, resulting in narrower margins and fewer support vectors (Hard Margin).

print("\n--- Step 3: Visualizing C Parameter Influence ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Create grid mesh for decision plotting
x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

for idx, c in enumerate(c_values):
    model_info = models[c]
    clf = model_info['model_obj']

    # Plot training points
    axes[idx].scatter(X_train_scaled[y_train == 0, 0], X_train_scaled[y_train == 0, 1], 
                      color='#e74c3c', label='Class 0', alpha=0.6, edgecolors='k')
    axes[idx].scatter(X_train_scaled[y_train == 1, 0], X_train_scaled[y_train == 1, 1], 
                      color='#3498db', label='Class 1', alpha=0.6, edgecolors='k')

    # Get decision function scores
    Z = clf.decision_function(grid_points).reshape(xx.shape)

    # Plot decision boundary (solid) and margins (dashed)
    axes[idx].contour(xx, yy, Z, levels=[-1, 0, 1], colors='black', 
                      linestyles=['--', '-', '--'], linewidths=[1, 2, 1])

    # Highlight support vectors
    axes[idx].scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=120,
                      facecolors='none', edgecolors='black', linewidths=1.5, label='Support Vectors')

    # Text settings
    axes[idx].set_title(f'C = {c} (SV Count: {model_info["support_vectors_count"]})\nAccuracy: {model_info["accuracy"]*100:.1f}%', 
                        fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Scaled X1')
    axes[idx].set_ylabel('Scaled X2')
    axes[idx].legend(loc='lower left')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Effect of Regularization Parameter (C) on SVM Margins', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()

plt.show()
print("Program completed successfully!")

