"""
svm_19_small_dataset_manual.py
------------------------------
Concept: Working with Tiny, Manually-Defined Datasets.
Algorithm: Support Vector Classifier (SVC) with a Linear Kernel.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Defines a tiny dataset (10 samples) directly in python lists.
2. Standardizes features manually/via scaler.
3. Fits a linear SVM.
4. Predicts class status for a new custom coordinate.
5. Displays a labeled scatter plot of the tiny dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


print("--- Step 1: Defining Tiny Manual Dataset ---")
# 10 samples of people: [Height in cm, Weight in kg]
X = np.array([
    [170, 55], [180, 85], [160, 50], [185, 95], [175, 70], 
    [150, 45], [165, 60], [190, 100], [155, 48], [172, 78]
])
# Target: 1 = Overweight or Heavy category, 0 = Normal/Light category
y = np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1])

df = pd.DataFrame(X, columns=['Height', 'Weight'])
df['Category'] = y
print("Tiny Dataset:")
print(df)

print("\n--- Step 2: Scaling Features ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled coordinates:")
print(X_scaled)

print("\n--- Step 3: Training the Model ---")
model = SVC(kernel='linear', C=1.0)
model.fit(X_scaled, y)
print("Model trained on 10 samples.")

print("\n--- Step 4: Predicting on a New Custom Point ---")
# Query patient: Height 178 cm, Weight 82 kg
new_patient = np.array([[178, 82]])
new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)

pred_label = "Heavy (1)" if prediction[0] == 1 else "Normal (0)"
print(f"New Patient Height: {new_patient[0,0]}cm, Weight: {new_patient[0,1]}kg")
print(f"Predicted Category: {pred_label}")

print("\n--- Step 5: Visualizing Tiny Dataset ---")
plt.figure(figsize=(7, 5))

# Plot manual coordinates
plt.scatter(df['Height'][y == 0], df['Weight'][y == 0], color='#3498db', s=100, label='Normal (0)', edgecolors='k')
plt.scatter(df['Height'][y == 1], df['Weight'][y == 1], color='#e74c3c', s=100, label='Heavy (1)', edgecolors='k')

# Plot the query patient
plt.scatter(new_patient[0, 0], new_patient[0, 1], color='#f1c40f', s=150, marker='*', label='New Query', edgecolors='k')

plt.title('SVM Classification on a Tiny Manual Dataset', fontsize=12, fontweight='bold')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

