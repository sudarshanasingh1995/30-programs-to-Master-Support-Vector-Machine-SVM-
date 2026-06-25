"""
svm_06_multiclass_digits.py
---------------------------
Concept: Multi-class Classification on Image Data.
Algorithm: Support Vector Classifier (SVC) with an RBF Kernel.
Dataset: Optical Recognition of Handwritten Digits.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Loads the handwritten digits dataset and filters it to classes 0, 1, and 2.
2. Displays digit representation in numerical shape.
3. Splits and scales features (pixels).
4. Trains an RBF SVM to classify the digit images.
5. Displays a grid plot displaying digits alongside predicted and actual labels.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


print("--- Step 1: Loading Handwritten Digits Dataset ---")
digits = load_digits()

# Filter dataset to digits 0, 1, and 2 for simplicity
mask = np.isin(digits.target, [0, 1, 2])
X = digits.data[mask]
y = digits.target[mask]
images = digits.images[mask]

print(f"Total images loaded: {len(X)}")
print(f"Image dimensions: {images[0].shape} pixels")
print("Numeric values for the first digit (pixel intensities 0-16):")
print(images[0])

print("\n--- Step 2: Preparing and Splitting Data ---")
# Split data into training and testing sets
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X, y, images, test_size=0.25, random_state=42, stratify=y
)

# Standardize the features (pixel intensities)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully.")

print("\n--- Step 3: Training RBF SVM ---")
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model.fit(X_train_scaled, y_train)
print("SVM digits classifier trained successfully.")

print("\n--- Step 4: Model Evaluation ---")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n--- Step 5: Visualizing Sample Predictions ---")
# Save a 2x4 grid plot of predictions
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 6))
axes = axes.ravel()

for i in range(8):
    axes[i].imshow(img_test[i], cmap='gray_r', interpolation='nearest')
    axes[i].set_title(f"True: {y_test[i]}\nPred: {y_pred[i]}", fontsize=10, 
                       fontweight='bold', color='green' if y_test[i] == y_pred[i] else 'red')
    axes[i].axis('off')

plt.suptitle('Digits Classification Predictions (0, 1, 2)', fontsize=14, fontweight='bold')
plt.tight_layout()

plt.show()
print("Program completed successfully!")

