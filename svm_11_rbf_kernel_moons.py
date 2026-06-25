"""
svm_11_rbf_kernel_moons.py
--------------------------
Concept: RBF Kernel on Moon-Shaped Data.
Algorithm: Support Vector Classifier (SVC) with an RBF Kernel.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates interlocking half-moon shapes using `make_moons`.
2. Scales features for distance balance.
3. Fits an SVM with RBF kernel to capture the curved boundary.
4. Evaluates test prediction performance.
5. Displays a filled contour plot representing decision regions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


print("--- Step 1: Generating Interlocking Moons ---")
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

df = pd.DataFrame(X, columns=['X1', 'X2'])
df['Label'] = y
print("Data sample:")
print(df.head())

print("\n--- Step 2: Preparing and Scaling Data ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data scaled.")

print("\n--- Step 3: Training RBF SVM ---")
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train_scaled, y_train)
print("RBF SVM trained successfully.")

print("\n--- Step 4: Model Evaluation ---")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n--- Step 5: Visualizing Filled Decision Regions ---")
plt.figure(figsize=(8, 6))

# Plot filled decision regions
xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-2.5, 2.5, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid_points).reshape(xx.shape)

# Filled contour of decision region colors
plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')

# Plot actual test coordinates
plt.scatter(X_test_scaled[y_test == 0, 0], X_test_scaled[y_test == 0, 1], 
            color='#3498db', label='Class 0', edgecolors='k')
plt.scatter(X_test_scaled[y_test == 1, 0], X_test_scaled[y_test == 1, 1], 
            color='#e74c3c', label='Class 1', edgecolors='k')

# Draw boundary line
Z_decision = model.decision_function(grid_points).reshape(xx.shape)
plt.contour(xx, yy, Z_decision, levels=[0], colors='black', linewidths=1.5)

plt.title('RBF SVM Decision Regions & Boundary (Moons)', fontsize=14, fontweight='bold')
plt.xlabel('Scaled X1')
plt.ylabel('Scaled X2')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

