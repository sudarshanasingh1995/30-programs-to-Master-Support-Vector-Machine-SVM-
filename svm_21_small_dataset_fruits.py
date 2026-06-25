"""
svm_21_small_dataset_fruits.py
------------------------------
Concept: Multi-class SVM on small hand-crafted features.
Algorithm: Support Vector Classifier (SVC) with an RBF Kernel.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Defines a small fruit dataset (12 samples) with mass (grams) and width (cm).
2. Classifies fruits into 3 categories: Apple (0), Orange (1), Lemon (2).
3. Standardizes features.
4. Trains an RBF SVM.
5. Performs classification validation and displays predictions.
6. Displays a scatter plot indicating fruit classifications.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


print("--- Step 1: Defining Hand-Crafted Fruit Dataset ---")
# Mass (g), Width (cm) for 12 fruits
X = np.array([
    [150, 7.2], [165, 7.5], [140, 7.0], # Apples (Class 0)
    [180, 8.0], [195, 8.5], [210, 9.0], # Oranges (Class 1)
    [110, 5.5], [120, 6.0], [105, 5.8], # Lemons (Class 2)
    [155, 7.3], [190, 8.2], [115, 5.9]  # Validation mix
])
# Labels: 0 = Apple, 1 = Orange, 2 = Lemon
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])

fruit_names = {0: 'Apple', 1: 'Orange', 2: 'Lemon'}
df = pd.DataFrame(X, columns=['Mass', 'Width'])
df['Fruit_Type'] = [fruit_names[i] for i in y]

print("Fruit Data Table:")
print(df)

print("\n--- Step 2: Scaling Features ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Mass and Width standardized to balance magnitude ranges.")

print("\n--- Step 3: Training RBF SVM ---")
# RBF kernel handles non-linear clusters
model = SVC(kernel='rbf', C=5.0, gamma='scale', random_state=42)
model.fit(X_scaled, y)
print("RBF SVM trained on fruit dataset.")

print("\n--- Step 4: Model Evaluation ---")
y_pred = model.predict(X_scaled)
acc = accuracy_score(y, y_pred)
print(f"Overall Accuracy: {acc * 100:.1f}%")

print("\nPredictions vs Actual:")
for idx, (actual, pred) in enumerate(zip(y, y_pred)):
    print(f" Fruit {idx+1:<2} | Actual: {fruit_names[actual]:<7} | Predicted: {fruit_names[pred]}")

print("\n--- Step 5: Visualizing Fruit Class Distributions ---")
plt.figure(figsize=(8, 6))

colors = ['#e74c3c', '#e67e22', '#f1c40f']
labels = ['Apple', 'Orange', 'Lemon']

for i in range(3):
    plt.scatter(df['Mass'][y == i], df['Width'][y == i], 
                color=colors[i], s=120, label=labels[i], edgecolors='k', alpha=0.9)

plt.title('Fruit Classification (Mass vs. Width)', fontsize=12, fontweight='bold')
plt.xlabel('Mass (grams)')
plt.ylabel('Width (cm)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

