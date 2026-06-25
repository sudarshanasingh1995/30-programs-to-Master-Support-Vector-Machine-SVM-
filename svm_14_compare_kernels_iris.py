"""
svm_14_compare_kernels_iris.py
------------------------------
Concept: Comparing SVM Kernels on Iris Dataset.
Algorithm: SVC with Linear, Polynomial, and RBF Kernels.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Loads the classic Iris flower dataset.
2. Standardizes features.
3. Fits three SVM classifiers using: Linear, Polynomial (degree 3), and RBF kernels.
4. Records training and test accuracies.
5. Displays a bar chart comparison to show which kernel fits the Iris classes best.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


print("--- Step 1: Loading Iris Dataset ---")
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Iris features loaded and standardized.")

print("\n--- Step 2: Training Kernels on Iris ---")
kernels = {
    'Linear': SVC(kernel='linear', C=1.0),
    'Polynomial (Deg 3)': SVC(kernel='poly', degree=3, C=1.0),
    'RBF': SVC(kernel='rbf', C=1.0, gamma='scale')
}

results = []

for name, clf in kernels.items():
    clf.fit(X_train_scaled, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))

    results.append({
        'Kernel': name,
        'Train_Accuracy': train_acc,
        'Test_Accuracy': test_acc
    })
    print(f"Kernel: {name:<20} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

results_df = pd.DataFrame(results)

print("\n--- Step 3: Visualizing Comparison Results ---")
plt.figure(figsize=(10, 6))

x = np.arange(len(results_df['Kernel']))
width = 0.35

plt.bar(x - width/2, results_df['Train_Accuracy'], width, label='Train Accuracy', color='#9b59b6', edgecolor='k')
plt.bar(x + width/2, results_df['Test_Accuracy'], width, label='Test Accuracy', color='#2ecc71', edgecolor='k')

plt.title('SVM Kernels on Iris: Train vs. Test Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Kernel Type')
plt.ylabel('Accuracy Score')
plt.xticks(x, results_df['Kernel'])
plt.ylim(0.8, 1.05)  # Zooms in on high accuracy range
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

for i, row in results_df.iterrows():
    plt.text(i - width/2, row['Train_Accuracy'] + 0.005, f"{row['Train_Accuracy'] * 100:.1f}%", ha='center', fontsize=9)
    plt.text(i + width/2, row['Test_Accuracy'] + 0.005, f"{row['Test_Accuracy'] * 100:.1f}%", ha='center', fontsize=9)

plt.show()
print("Program completed successfully!")

