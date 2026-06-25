"""
svm_28_compare_knn_logistic_blobs.py
------------------------------------
Concept: Comparing SVM, KNN, and Logistic Regression on Separable Data.
Algorithm: Linear SVM vs. KNeighborsClassifier vs. LogisticRegression.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates a linearly separable synthetic 2D binary dataset.
2. Standardizes coordinates.
3. Fits three distinct classifiers: Linear SVM, KNN (K=5), and Logistic Regression.
4. Records and prints test accuracies.
5. Displays a 3-panel side-by-side plot comparing decision boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


print("--- Step 1: Generating Linearly Separable Blobs ---")
X, y = make_blobs(n_samples=120, centers=2, n_features=2, cluster_std=1.0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data standardized.")

print("\n--- Step 2: Training the Three Classifiers ---")
classifiers = {
    'Linear SVM': SVC(kernel='linear', C=1.0),
    'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression()
}

results = {}

for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    results[name] = {
        'model_obj': clf,
        'accuracy': acc
    }
    print(f"Classifier: {name:<20} | Test Accuracy: {acc*100:.1f}%")

print("\n--- Step 3: Visualizing and Comparing Boundaries ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Create grid mesh for decision plotting
x_min, x_max = X_train_scaled[:, 0].min() - 0.5, X_train_scaled[:, 0].max() + 0.5
y_min, y_max = X_train_scaled[:, 1].min() - 0.5, X_train_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

for idx, (name, clf_info) in enumerate(results.items()):
    clf = clf_info['model_obj']

    # Predict regions
    Z = clf.predict(grid_points).reshape(xx.shape)
    axes[idx].contourf(xx, yy, Z, alpha=0.15, cmap='coolwarm')

    # Plot test data points
    axes[idx].scatter(X_test_scaled[y_test == 0, 0], X_test_scaled[y_test == 0, 1], 
                      color='#e74c3c', label='Class 0', edgecolors='k')
    axes[idx].scatter(X_test_scaled[y_test == 1, 0], X_test_scaled[y_test == 1, 1], 
                      color='#3498db', label='Class 1', edgecolors='k')

    # Add decision boundaries
    if hasattr(clf, "decision_function"):
        Z_line = clf.decision_function(grid_points).reshape(xx.shape)
        axes[idx].contour(xx, yy, Z_line, levels=[0], colors='black', linewidths=2)
    elif hasattr(clf, "predict_proba"):
        Z_line = clf.predict_proba(grid_points)[:, 1].reshape(xx.shape)
        axes[idx].contour(xx, yy, Z_line, levels=[0.5], colors='black', linewidths=2)

    axes[idx].set_title(f'{name}\nAccuracy: {clf_info["accuracy"]*100:.1f}%', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Scaled X1')
    axes[idx].set_ylabel('Scaled X2')
    axes[idx].legend(loc='best')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Comparison of SVM, KNN, and Logistic Regression on Separable Data', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()

plt.show()
print("Program completed successfully!")

