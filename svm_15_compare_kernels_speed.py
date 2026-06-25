"""
svm_15_compare_kernels_speed.py
-------------------------------
Concept: Comparing SVM Kernel Computational Speeds.
Algorithm: SVC with Linear, Polynomial, and RBF Kernels.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates a moderately large synthetic dataset (1,500 samples).
2. Fits Linear, Polynomial, and RBF SVM models.
3. Measures the training (fit) time and prediction (inference) time in milliseconds.
4. Explains the computational complexity trade-offs of kernel methods.
5. Displays a bar chart illustrating training speeds.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


print("--- Step 1: Generating Moderately Large Dataset ---")
# 1,500 samples to make timing difference visible
X, y = make_classification(n_samples=1500, n_features=20, n_informative=15, 
                           n_classes=2, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Data shape: {X_scaled.shape}")

print("\n--- Step 2: Timing the Fitting & Prediction of Different Kernels ---")
kernels = {
    'Linear': SVC(kernel='linear', C=1.0),
    'Polynomial (Deg 3)': SVC(kernel='poly', degree=3, C=1.0),
    'RBF': SVC(kernel='rbf', C=1.0, gamma='scale')
}

results = []

for name, clf in kernels.items():
    print(f"Benchmarking {name} kernel...")

    # Measure Training Time
    start_fit = time.time()
    clf.fit(X_scaled, y)
    end_fit = time.time()
    fit_time = (end_fit - start_fit) * 1000 # in milliseconds

    # Measure Prediction Time
    start_pred = time.time()
    _ = clf.predict(X_scaled)
    end_pred = time.time()
    pred_time = (end_pred - start_pred) * 1000 # in milliseconds

    results.append({
        'Kernel': name,
        'Training_Time_ms': fit_time,
        'Prediction_Time_ms': pred_time
    })
    print(f" -> Training time: {fit_time:.2f} ms | Prediction time: {pred_time:.2f} ms")

results_df = pd.DataFrame(results)

print("\n--- Step 3: Visualizing Speed Differences ---")
plt.figure(figsize=(10, 6))

x = np.arange(len(results_df['Kernel']))
width = 0.35

plt.bar(x - width/2, results_df['Training_Time_ms'], width, label='Training Time (ms)', color='#e74c3c', edgecolor='k')
plt.bar(x + width/2, results_df['Prediction_Time_ms'], width, label='Prediction Time (ms)', color='#34495e', edgecolor='k')

plt.title('SVM Computational Time Comparison (Lower is Faster)', fontsize=14, fontweight='bold')
plt.xlabel('Kernel Type')
plt.ylabel('Time in Milliseconds')
plt.xticks(x, results_df['Kernel'])
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

for i, row in results_df.iterrows():
    plt.text(i - width/2, row['Training_Time_ms'] + (row['Training_Time_ms']*0.02 + 0.1), f"{row['Training_Time_ms']:.1f}ms", ha='center', fontsize=9)
    plt.text(i + width/2, row['Prediction_Time_ms'] + (row['Prediction_Time_ms']*0.02 + 0.1), f"{row['Prediction_Time_ms']:.1f}ms", ha='center', fontsize=9)

plt.show()
print("Program completed successfully!")

