"""
svm_02_binary_health_prediction.py
----------------------------------
Concept: Binary Classification (Yes/No) on Healthcare Data.
Algorithm: Support Vector Classifier (SVC) with an RBF Kernel.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates a synthetic dataset of patient medical metrics (e.g., Blood Pressure & Age).
2. Sets a classification target representing "Healthy" vs. "High Risk" of diabetes.
3. Splits, scales, and trains an SVM classifier with a Radial Basis Function (RBF) kernel.
4. Outputs the model accuracy and classification report.
5. Displays a scatter plot showing patient risk levels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


print("--- Step 1: Simulating Patient Health Data ---")
np.random.seed(42)

# Generate synthetic health features for 200 patients
# Feature 1: Age (range 20 to 80)
# Feature 2: Systolic Blood Pressure (range 90 to 180 mmHg)
age = np.random.uniform(20, 80, size=200)
blood_pressure = np.random.uniform(90, 180, size=200)

df = pd.DataFrame({
    'Age': age,
    'Blood_Pressure': blood_pressure
})

# Define risk rule: High risk if age and blood pressure are high (non-linear threshold)
# Formula creates a curve: risk is higher for older people even with lower BP, and vice versa.
risk_score = (df['Age'] - 45) * 1.5 + (df['Blood_Pressure'] - 120) * 2.0
df['Risk_Status'] = (risk_score > 10).astype(int)  # 1 = High Risk, 0 = Healthy

print("Sample Patient Records:")
print(df.head())
print(f"\nTarget distribution:\n{df['Risk_Status'].value_counts()}")

print("\n--- Step 2: Preparing Features and Splitting ---")
X = df[['Age', 'Blood_Pressure']]
y = df['Risk_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Patient data scaled successfully.")

print("\n--- Step 3: Training RBF SVM ---")
# RBF kernel handles non-linear relationships.
# C controls trade-off between smooth boundary and classifying training points correctly.
# gamma controls the radius of influence of a single training point.
model = SVC(kernel='rbf', C=1.5, gamma='scale')
model.fit(X_train_scaled, y_train)
print("RBF SVM model trained successfully.")

print("\n--- Step 4: Model Evaluation ---")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'High Risk']))

print("\n--- Step 5: Visualizing Results ---")
plt.figure(figsize=(8, 6))

# Plot patients: Healthy (green) vs High Risk (red)
plt.scatter(X_test['Age'][y_test == 0], X_test['Blood_Pressure'][y_test == 0], 
            color='#2ecc71', label='Healthy (0)', edgecolors='k', alpha=0.8)
plt.scatter(X_test['Age'][y_test == 1], X_test['Blood_Pressure'][y_test == 1], 
            color='#e74c3c', label='High Risk (1)', edgecolors='k', alpha=0.8)

plt.title('SVM Classification of Patient Risk Levels', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Systolic Blood Pressure (mmHg)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

