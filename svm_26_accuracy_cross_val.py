"""
svm_26_accuracy_cross_val.py
----------------------------
Concept: Cross Validation for Model Stability.
Algorithm: Support Vector Classifier (SVC) with Cross Validation.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates a synthetic classification dataset.
2. Standardizes features.
3. Defines a Support Vector Machine classifier.
4. Performs 5-Fold Cross Validation (`cross_val_score`).
5. Prints validation accuracy for each fold, along with mean and standard deviation.
6. Displays a bar chart displaying fold accuracies against the mean performance baseline.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


print("--- Step 1: Generating and Preparing Data ---")
X, y = make_classification(n_samples=250, n_features=6, n_informative=4, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Dataset generated and scaled.")

print("\n--- Step 2: Setting up the SVM Classifier ---")
# Define RBF SVM
model = SVC(kernel='rbf', C=1.0)

print("\n--- Step 3: Running 5-Fold Cross Validation ---")
# cross_val_score splits the data, fits the model, and scores it 5 separate times.
# This prevents bias from a single train-test split.
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

print("Cross-Validation Scores for each fold:")
for fold, score in enumerate(scores, 1):
    print(f" Fold {fold}: {score:.4f} ({score * 100:.2f}%)")

mean_score = scores.mean()
std_dev = scores.std()

print(f"\nSummary Performance Metrics:")
print(f" Mean Accuracy:              {mean_score:.4f} ({mean_score * 100:.2f}%)")
print(f" Standard Deviation (std):   {std_dev:.4f} (Indicates variation between folds)")
print(f" 95% Confidence Interval:    ({mean_score - 2*std_dev:.4f} to {mean_score + 2*std_dev:.4f})")

print("\n--- Step 4: Visualizing Fold Accuracies ---")
plt.figure(figsize=(8, 5))

folds = [f"Fold {i}" for i in range(1, 6)]
plt.bar(folds, scores, color='#3498db', edgecolor='k', alpha=0.8, width=0.5, label='Fold Accuracy')

# Draw a horizontal line at the mean accuracy level
plt.axhline(mean_score, color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean Accuracy ({mean_score*100:.1f}%)')

plt.title('5-Fold Cross Validation Accuracies', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3, axis='y')

# Annotate accuracy values on top of bars
for idx, val in enumerate(scores):
    plt.text(idx, val + 0.02, f"{val*100:.1f}%", ha='center', fontsize=9, fontweight='bold')

plt.show()
print("Program completed successfully!")

