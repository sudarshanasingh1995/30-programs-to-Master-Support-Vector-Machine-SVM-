"""
svm_23_tune_gamma.py
--------------------
Concept: Tuning the RBF Kernel Bandwidth (gamma).
Algorithm: Support Vector Classifier (SVC) with an RBF Kernel.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates a complex non-linear classification dataset (half-moons).
2. Fits three separate RBF SVMs with different gamma values: gamma=0.1, 1.0, 10.0 (holding C=1.0 constant).
3. Compares and prints the training vs. test accuracy for each model.
4. Displays a 3-panel visualization showing decision boundaries to contrast underfitting vs. overfitting.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


print("--- Step 1: Generating Non-Linear Data (Moons) ---")
X, y = make_moons(n_samples=150, noise=0.25, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data scaled successfully.")

print("\n--- Step 2: Training SVMs with Different Gamma Values ---")
gamma_values = [0.1, 1.0, 10.0]
models = {}

for g in gamma_values:
    # Train RBF SVM
    model = SVC(kernel='rbf', C=1.0, gamma=g)
    model.fit(X_train_scaled, y_train)

    # Check accuracies
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

    models[g] = {
        'model_obj': model,
        'train_acc': train_acc,
        'test_acc': test_acc
    }

    print(f"RBF SVM with gamma = {g:<4} | Train Accuracy: {train_acc*100:.1f}% | Test Accuracy: {test_acc*100:.1f}%")
    # Low gamma values: Smooth, broad boundaries (underfitting risk).
    # High gamma values: Tight, wiggly boundaries enclosing individual samples (overfitting risk).

print("\n--- Step 3: Visualizing Gamma Parameter Influence ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Create grid mesh for decision plotting
x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

for idx, g in enumerate(gamma_values):
    model_info = models[g]
    clf = model_info['model_obj']

    # Color decision region
    Z = clf.decision_function(grid_points).reshape(xx.shape)
    axes[idx].contourf(xx, yy, Z > 0, alpha=0.15, cmap='coolwarm')

    # Plot training points
    axes[idx].scatter(X_train_scaled[y_train == 0, 0], X_train_scaled[y_train == 0, 1], 
                      color='#e74c3c', label='Class 0', alpha=0.7, edgecolors='k')
    axes[idx].scatter(X_train_scaled[y_train == 1, 0], X_train_scaled[y_train == 1, 1], 
                      color='#3498db', label='Class 1', alpha=0.7, edgecolors='k')

    # Draw boundary contours where decision score = 0
    axes[idx].contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

    # Text settings
    axes[idx].set_title(f'gamma = {g}\nTrain: {model_info["train_acc"]*100:.1f}% | Test: {model_info["test_acc"]*100:.1f}%', 
                        fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Scaled X1')
    axes[idx].set_ylabel('Scaled X2')
    axes[idx].legend(loc='lower left')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Effect of RBF Kernel Gamma on Boundary Complexity', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()

plt.show()
print("Program completed successfully!")

