"""
svm_07_linear_kernel_salary.py
------------------------------
Concept: Linear Kernel SVM.
Algorithm: Support Vector Classifier (SVC) with a Linear Kernel.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Simulates salary class data (High vs. Low) based on experience and education score.
2. Standardizes features for distance scaling consistency.
3. Fits an SVM with a linear kernel.
4. Evaluates the classifier's accuracy.
5. Displays a scatter plot indicating the linear boundary separating salary brackets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


print("--- Step 1: Simulating Salary and Experience Data ---")
np.random.seed(42)

# Generate synthetic details for 150 employees
# Feature 1: Years of experience (1 to 20 years)
# Feature 2: Education score (1 = High School, 5 = PhD)
experience = np.random.uniform(1, 20, size=150)
education_score = np.random.uniform(1, 5, size=150)

df = pd.DataFrame({
    'Experience': experience,
    'Education_Score': education_score
})

# Salary rule: High salary (1) if Experience * 2.5 + Education_Score * 8 > 28
df['High_Salary'] = ((df['Experience'] * 2.5 + df['Education_Score'] * 8.0) > 28).astype(int)

print("Sample Employee Profiles:")
print(df.head())
print(f"\nSalary Classes (0 = Low, 1 = High):\n{df['High_Salary'].value_counts()}")

print("\n--- Step 2: Preparing and Scaling Data ---")
X = df[['Experience', 'Education_Score']]
y = df['High_Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features standardized.")

print("\n--- Step 3: Training Linear SVM Model ---")
# Train a model enforcing a linear decision boundary
model = SVC(kernel='linear', C=1.0)
model.fit(X_train_scaled, y_train)
print("Linear kernel model fitting complete.")

print("\n--- Step 4: Model Evaluation ---")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Salary', 'High Salary']))

print("\n--- Step 5: Visualizing Linear Boundary ---")
plt.figure(figsize=(8, 6))

# Plot employee records
plt.scatter(X_test['Experience'][y_test == 0], X_test['Education_Score'][y_test == 0], 
            color='#95a5a6', label='Low Salary (0)', edgecolors='k', alpha=0.8)
plt.scatter(X_test['Experience'][y_test == 1], X_test['Education_Score'][y_test == 1], 
            color='#2ecc71', label='High Salary (1)', edgecolors='k', alpha=0.8)

plt.title('Linear Decision Boundary for Salary Classification', fontsize=14, fontweight='bold')
plt.xlabel('Years of Experience')
plt.ylabel('Education Score (1-5)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

