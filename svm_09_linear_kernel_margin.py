"""
svm_09_linear_kernel_margin.py
------------------------------
Concept: Linear Kernel Math & Margin Analysis.
Algorithm: Support Vector Classifier (SVC) with a Linear Kernel.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates a small, linearly separable 2D dataset.
2. Trains a linear SVM model.
3. Explains and extracts hyperplane coefficients (w) and intercept (b).
4. Highlights and prints the coordinates of the support vectors.
5. Calculates the exact width of the margins mathematically (2 / ||w||).
6. Displays a plot of the hyperplane, highlighting the support vectors with text annotations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC


print("--- Step 1: Defining 2D Separable Points ---")
# Hardcoded dataset for clarity
X = np.array([
    [2, 3], [1, 2], [2, 1], [3, 2],
    [5, 6], [6, 5], [5, 4], [7, 6]
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

df = pd.DataFrame(X, columns=['X1', 'X2'])
df['Label'] = y
print("Data Points:")
print(df)

print("\n--- Step 2: Training the Linear SVM ---")
# Use a strict C value to avoid errors in training (hard margin approximation)
model = SVC(kernel='linear', C=10.0)
model.fit(X, y)
print("Model trained.")

print("\n--- Step 3: Extracting Hyperplane Math Parameters ---")
w = model.coef_[0]
b = model.intercept_[0]
print(f"Weight vector w = {w}")
print(f"Intercept b = {b}")
print(f"Mathematical formula of Hyperplane: {w[0]:.2f}*x1 + {w[1]:.2f}*x2 + {b:.2f} = 0")

print("\n--- Step 4: Accessing Support Vectors ---")
support_vectors = model.support_vectors_
print(f"Number of support vectors found: {len(support_vectors)}")
print("Coordinates of Support Vectors:")
for idx, sv in enumerate(support_vectors):
    print(f" SV {idx+1}: {sv}")

print("\n--- Step 5: Calculating Margin Width ---")
# The margin distance is given by 2 / ||w|| where ||w|| is Euclidean norm.
w_norm = np.linalg.norm(w)
margin_width = 2.0 / w_norm
print(f"Euclidean norm of w: {w_norm:.4f}")
print(f"Calculated Margin Width (distance between dashed lines): {margin_width:.4f}")

print("\n--- Step 6: Visualizing the Margins & Hyperplane ---")
plt.figure(figsize=(8, 8))

# Plot data points
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='#e74c3c', s=100, label='Class 0', edgecolors='k')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='#3498db', s=100, label='Class 1', edgecolors='k')

# Draw the decision hyperplane and margins
ax = plt.gca()
xlim = (0, 8)
ylim = (0, 8)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

xx = np.linspace(xlim[0], xlim[1], 100)

# Decision boundary: w0*x1 + w1*x2 + b = 0  => x2 = -(w0*x1 + b) / w1
yy_decision = -(w[0] * xx + b) / w[1]
# Upper margin: w0*x1 + w1*x2 + b = 1     => x2 = -(w0*x1 + b - 1) / w1
yy_upper = -(w[0] * xx + b - 1) / w[1]
# Lower margin: w0*x1 + w1*x2 + b = -1    => x2 = -(w0*x1 + b + 1) / w1
yy_lower = -(w[0] * xx + b + 1) / w[1]

# Plot lines
plt.plot(xx, yy_decision, 'k-', linewidth=2, label='Hyperplane (wTx + b = 0)')
plt.plot(xx, yy_upper, 'k--', label='Margin Boundary (wTx + b = 1)')
plt.plot(xx, yy_lower, 'k--', label='Margin Boundary (wTx + b = -1)')

# Highlight support vectors and annotate coordinates
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=250,
           linewidth=2, facecolors='none', edgecolors='green', label='Support Vectors')

for sv in support_vectors:
    plt.annotate(f"({sv[0]},{sv[1]})", (sv[0]+0.15, sv[1]-0.15), fontsize=10, fontweight='bold', color='green')

plt.title('SVM Math: Hyperplane, Support Vectors & Margin Width', fontsize=14, fontweight='bold')
plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

