"""
svm_01_binary_separable.py
--------------------------
Concept: Binary Classification (Yes/No) with Linearly Separable Data.
Algorithm: Support Vector Classifier (SVC) with a Linear Kernel.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates a synthetic, linearly separable binary classification dataset.
2. Standardizes features for optimal SVM performance.
3. Fits a linear SVM model to classify the points.
4. Evaluates the classifier's accuracy.
5. Displays a scatter plot showing the decision boundary and margins.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


print("--- Step 1: Generating Linearly Separable Data ---")
# Generate 150 samples with 2 features and 2 distinct clusters
X, y = make_blobs(n_samples=150, centers=2, n_features=2, random_state=42, cluster_std=1.2)

# Convert to DataFrame for readability
df = pd.DataFrame(X, columns=['Feature_A', 'Feature_B'])
df['Label'] = y

print("Sample Dataset:")
print(df.head())
print(f"Total samples: {len(df)} (Class 0: {sum(y==0)}, Class 1: {sum(y==1)})")

print("\n--- Step 2: Splitting and Scaling Features ---")
# SVM is distance-based, so standardizing features is a best practice.
X_train, X_test, y_train, y_test = train_test_split(
    df[['Feature_A', 'Feature_B']], df['Label'], test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled to have zero mean and unit variance.")

print("\n--- Step 3: Training the SVM Classifier (Linear Kernel) ---")
# C is the regularization parameter. Large C means strict classification (small margin).
model = SVC(kernel='linear', C=1.0)
model.fit(X_train_scaled, y_train)
print("Model training complete.")

print("\n--- Step 4: Model Evaluation ---")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n--- Step 5: Visualizing the Decision Boundary ---")
plt.figure(figsize=(8, 6))

# Plot training data points
plt.scatter(X_train_scaled[y_train == 0, 0], X_train_scaled[y_train == 0, 1], 
            color='#e74c3c', label='Class 0', alpha=0.7, edgecolors='k')
plt.scatter(X_train_scaled[y_train == 1, 0], X_train_scaled[y_train == 1, 1], 
            color='#3498db', label='Class 1', alpha=0.7, edgecolors='k')

# Create a mesh grid to plot the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

# Plot decision boundary (Z=0) and margins (Z=-1, Z=1)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Highlight support vectors
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='black', label='Support Vectors')

plt.title('SVM Linear Decision Boundary & Margins', fontsize=14, fontweight='bold')
plt.xlabel('Scaled Feature A')
plt.ylabel('Scaled Feature B')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

