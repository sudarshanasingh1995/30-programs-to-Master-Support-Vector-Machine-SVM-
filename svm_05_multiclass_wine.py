"""
svm_05_multiclass_wine.py
-------------------------
Concept: Multi-class Classification (3 Classes).
Algorithm: Support Vector Classifier (SVC) with an RBF Kernel.
Dataset: Wine Recognition Dataset.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Loads the Wine dataset from scikit-learn (representing chemical profiles of wines).
2. Performs data splitting and standardizes the features.
3. Trains a Support Vector Machine classifier.
4. Generates a classification report and evaluates accuracy.
5. Displays a scatter plot showing wine class classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


print("--- Step 1: Loading Wine Dataset ---")
wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

# Convert to DataFrame to print sample statistics
df = pd.DataFrame(X, columns=wine.feature_names)
df['Wine_Class'] = [target_names[i] for i in y]

print("Wine Data Sample (Features relate to chemical analysis):")
print(df.head())
print(f"\nTarget classes: {target_names} (Distribution: {np.bincount(y)})")

print("\n--- Step 2: Preparing and Splitting the Data ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully.")

print("\n--- Step 3: Training the SVM Classifier ---")
# Using a Radial Basis Function (RBF) kernel to classify the 13-feature wine chemical space
model = SVC(kernel='rbf', C=2.0, gamma='scale', random_state=42)
model.fit(X_train_scaled, y_train)
print("RBF SVM trained on wine dataset.")

print("\n--- Step 4: Model Evaluation ---")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("\n--- Step 5: Visualizing Class Separability ---")
# Plot Alcohol vs. Malic Acid (first 2 features of the wine dataset)
plt.figure(figsize=(8, 6))

colors = ['#9b59b6', '#34495e', '#e67e22']
for i, color in enumerate(colors):
    plt.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1], 
                color=color, label=target_names[i], edgecolors='k', s=60, alpha=0.8)

plt.title('Wine Classification (Alcohol vs. Malic Acid)', fontsize=14, fontweight='bold')
plt.xlabel(wine.feature_names[0].capitalize())
plt.ylabel(wine.feature_names[1].capitalize())
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

