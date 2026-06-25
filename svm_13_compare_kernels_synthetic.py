"""
svm_13_compare_kernels_synthetic.py
-----------------------------------
Concept: Comparing SVM Kernels on Synthetic Data.
Algorithm: SVC with Linear, Polynomial, and RBF Kernels.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates a noisy synthetic binary classification dataset.
2. Standardizes features.
3. Fits three SVM classifiers using: Linear, Polynomial (degree 3), and RBF kernels.
4. Records training and test set accuracies for each model.
5. Displays a comparison bar chart showing kernel generalization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


print("--- Step 1: Generating Noisy Synthetic Data ---")
# 200 samples, 2 features, noisy configuration
X, y = make_classification(n_samples=250, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, flip_y=0.15, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Dataset generated and scaled.")

print("\n--- Step 2: Training SVM with Different Kernels ---")
kernels = {
    'Linear': SVC(kernel='linear', C=1.0),
    'Polynomial (Deg 3)': SVC(kernel='poly', degree=3, C=1.0),
    'RBF': SVC(kernel='rbf', C=1.0, gamma='scale')
}

results = []

for name, clf in kernels.items():
    # Fit the classifier
    clf.fit(X_train_scaled, y_train)

    # Calculate train and test accuracy
    train_pred = clf.predict(X_train_scaled)
    test_pred = clf.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    results.append({
        'Kernel': name,
        'Train_Accuracy': train_acc,
        'Test_Accuracy': test_acc
    })
    print(f"Kernel: {name:<20} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# Convert results list to DataFrame
results_df = pd.DataFrame(results)

print("\n--- Step 3: Visualizing Kernel Accuracy ---")
# Plot comparing Train vs Test accuracy for each kernel
plt.figure(figsize=(10, 6))

x = np.arange(len(results_df['Kernel']))
width = 0.35

plt.bar(x - width/2, results_df['Train_Accuracy'], width, label='Train Accuracy', color='#3498db', edgecolor='k')
plt.bar(x + width/2, results_df['Test_Accuracy'], width, label='Test Accuracy', color='#e67e22', edgecolor='k')

plt.title('SVM Kernel Comparison: Train vs. Test Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Kernel Type')
plt.ylabel('Accuracy Score')
plt.xticks(x, results_df['Kernel'])
plt.ylim(0, 1.1)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# Add scores above bars
for i, row in results_df.iterrows():
    plt.text(i - width/2, row['Train_Accuracy'] + 0.02, f"{row['Train_Accuracy'] * 100:.1f}%", ha='center', fontsize=9)
    plt.text(i + width/2, row['Test_Accuracy'] + 0.02, f"{row['Test_Accuracy'] * 100:.1f}%", ha='center', fontsize=9)

plt.show()
print("Program completed successfully!")

