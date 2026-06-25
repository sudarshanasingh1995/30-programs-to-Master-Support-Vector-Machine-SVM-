"""
svm_30_compare_knn_logistic_real.py
-----------------------------------
Concept: Comparing SVM, KNN, and Logistic Regression on Real-World Data.
Algorithm: RBF SVM vs. KNeighborsClassifier vs. LogisticRegression.
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Loads the real-world Breast Cancer dataset from scikit-learn.
2. Splits data and standardizes features.
3. Fits three classifiers: RBF SVM, KNN (K=5), and Logistic Regression.
4. Measures training (fit) time and test accuracy for each.
5. Prints a performance summary table.
6. Displays a comparison bar chart displaying both metrics side-by-side.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


print("--- Step 1: Loading Breast Cancer Dataset ---")
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
target_names = cancer.target_names

print(f"Dataset dimensions: {X.shape} (30 features, {X.shape[0]} samples)")
print(f"Target classes: {target_names} (Distribution: {np.bincount(y)})")

print("\n--- Step 2: Preparing and Scaling Data ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data standardized.")

print("\n--- Step 3: Benchmarking the Three Classifiers ---")
classifiers = {
    'RBF SVM': SVC(kernel='rbf', C=1.0, random_state=42),
    'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

results = []

for name, clf in classifiers.items():
    # Measure Training Time
    start_time = time.time()
    clf.fit(X_train_scaled, y_train)
    end_time = time.time()
    fit_time = (end_time - start_time) * 1000 # in milliseconds

    # Measure Test Accuracy
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    results.append({
        'Classifier': name,
        'Test_Accuracy': acc,
        'Training_Time_ms': fit_time
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

print("\nPerformance Comparison Summary Table:")
print(results_df.to_string(index=False))

print("\n--- Step 4: Visualizing Accuracy and Speed Trade-offs ---")
fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(results_df['Classifier']))
width = 0.35

# Plot Accuracy on primary Y-axis (left)
color_acc = '#2ecc71'
rects1 = ax1.bar(x - width/2, results_df['Test_Accuracy'], width, label='Test Accuracy', color=color_acc, edgecolor='k')
ax1.set_xlabel('Classifier Type')
ax1.set_ylabel('Accuracy Score', color='black')
ax1.set_ylim(0.9, 1.01) # Zoom in on the high accuracies
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['Classifier'])

# Create a secondary Y-axis (right) for training times
ax2 = ax1.twinx()
color_time = '#e74c3c'
rects2 = ax2.bar(x + width/2, results_df['Training_Time_ms'], width, label='Training Time (ms)', color=color_time, edgecolor='k')
ax2.set_ylabel('Training Time (ms)', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Label values on top of the bars
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f"{h*100:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax2.annotate(f"{h:.1f} ms", xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

plt.title('Classifier Accuracy vs. Training Speed on Breast Cancer Dataset', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

