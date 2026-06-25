"""
svm_04_multiclass_iris.py
-------------------------
Concept: Multi-class Classification (3 Classes).
Algorithm: Support Vector Classifier (SVC) with an RBF Kernel.
Dataset: Classical Iris Flower Dataset.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Loads the Iris dataset from scikit-learn (3 target classes).
2. Splits and normalizes the feature metrics (Sepal and Petal details).
3. Configures an SVC model (implicitly using One-vs-One decision strategy for multi-class).
4. Trains the model and predicts the classes.
5. Displays a scatter plot highlighting class boundaries using Sepal measurements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


print("--- Step 1: Loading Iris Dataset ---")
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Convert to DataFrame for visualization of structure
df = pd.DataFrame(X, columns=iris.feature_names)
df['Species'] = [target_names[i] for i in y]

print("Iris Data Sample:")
print(df.head())
print(f"\nTarget classes: {target_names} (Distribution: {np.bincount(y)})")

print("\n--- Step 2: Preparing and Splitting the Data ---")
# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling is critical for SVM to work correctly across diverse feature ranges.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features standardized.")

print("\n--- Step 3: Training the Multi-Class SVM ---")
# By default, SVC uses the One-Vs-One (OvO) approach for multi-class classification.
# For N classes, it trains N*(N-1)/2 binary classifiers.
model = SVC(kernel='rbf', C=1.0, decision_function_shape='ovo', random_state=42)
model.fit(X_train_scaled, y_train)
print("Multi-class OvO RBF SVM trained.")

print("\n--- Step 4: Model Evaluation ---")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("\n--- Step 5: Visualizing Class Separability ---")
# We will plot Sepal Length vs Sepal Width (the first two features) to visualize classes.
plt.figure(figsize=(8, 6))

colors = ['#e74c3c', '#3498db', '#2ecc71']
for i, color in enumerate(colors):
    plt.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1], 
                color=color, label=target_names[i], edgecolors='k', s=60, alpha=0.8)

plt.title('Iris Flower Species Classification (Sepal Features)', fontsize=14, fontweight='bold')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

