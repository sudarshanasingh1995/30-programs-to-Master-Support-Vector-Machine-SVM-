"""
svm_25_accuracy_confusion_matrix.py
-----------------------------------
Concept: Model Evaluation - Accuracy & Confusion Matrix.
Algorithm: Support Vector Classifier (SVC) with an RBF Kernel.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates a synthetic binary classification dataset.
2. Fits an RBF SVM model.
3. Computes standard metrics: Accuracy, Precision, Recall, and F1-Score.
4. Generates the numeric Confusion Matrix.
5. Displays a colored Confusion Matrix Grid visualization using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, ConfusionMatrixDisplay
)


print("--- Step 1: Generating and Preparing Data ---")
# 150 samples with 2 classes
X, y = make_classification(n_samples=180, n_features=4, n_informative=3, n_redundant=0, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data standardized.")

print("\n--- Step 2: Training the SVM Classifier ---")
model = SVC(kernel='rbf', C=1.0, random_state=42)
model.fit(X_train_scaled, y_train)
print("RBF SVM classifier trained successfully.")

print("\n--- Step 3: Computing Performance Metrics ---")
y_pred = model.predict(X_test_scaled)

# Metrics calculation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy Score:  {accuracy:.4f} ({accuracy * 100:.1f}%)")
print(f"Precision Score: {precision:.4f} (Of all predicted positive, how many are actually positive)")
print(f"Recall Score:    {recall:.4f} (Of all actual positive, how many were captured)")
print(f"F1 Score:        {f1:.4f} (Harmonic mean of precision and recall)")

print("\n--- Step 4: Computing Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix Array:")
print(cm)
print(f"True Negatives (TN): {cm[0,0]} | False Positives (FP): {cm[0,1]}")
print(f"False Negatives (FN): {cm[1,0]} | True Positives (TP): {cm[1,1]}")

print("\n--- Step 5: Visualizing and Saving Confusion Matrix ---")
plt.figure(figsize=(6, 6))

# Plot confusion matrix using sklearn display helper
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap='Blues', ax=plt.gca(), colorbar=False)

plt.title('SVM Classification Confusion Matrix', fontsize=12, fontweight='bold')
plt.grid(False) # Turn off layout grid lines for clearer heatmap boxes

plt.show()
print("Program completed successfully!")

