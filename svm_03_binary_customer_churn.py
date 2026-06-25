"""
svm_03_binary_customer_churn.py
--------------------------------
Concept: Binary Classification (Yes/No) for Customer Churn.
Algorithm: Support Vector Classifier (SVC) with a Linear Kernel.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates a synthetic customer dataset with transaction counts and support interactions.
2. Labels customers who churned vs. those who remained loyal.
3. Scales features and trains a linear SVM classifier.
4. Reports the accuracy and predictions.
5. Displays a scatter plot showing customer churn status.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


print("--- Step 1: Simulating Customer Data ---")
np.random.seed(42)

# Generate synthetic features for 180 customers
# Feature 1: Monthly transaction volume (1 to 60 transactions)
# Feature 2: Number of support calls (0 to 10 calls)
transactions = np.random.randint(1, 60, size=180)
support_calls = np.random.randint(0, 10, size=180)

df = pd.DataFrame({
    'Transactions': transactions,
    'Support_Calls': support_calls
})

# Define Churn rule: Customer is likely to churn (1) if support calls are high
# and transaction volume is low.
churn_score = (df['Support_Calls'] * 6) - df['Transactions']
df['Churned'] = (churn_score > 5).astype(int)  # 1 = Churned, 0 = Loyal

print("Sample Customer Records:")
print(df.head())
print(f"\nTarget distribution (0 = Loyal, 1 = Churned):\n{df['Churned'].value_counts()}")

print("\n--- Step 2: Preparing Features and Splitting ---")
X = df[['Transactions', 'Support_Calls']]
y = df['Churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Customer features standardized.")

print("\n--- Step 3: Training the SVM Model ---")
# Use a linear kernel to model simple boundary
model = SVC(kernel='linear', C=1.0)
model.fit(X_train_scaled, y_train)
print("Linear SVM trained successfully.")

print("\n--- Step 4: Model Evaluation ---")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Loyal', 'Churned']))

print("\n--- Step 5: Visualizing Results ---")
plt.figure(figsize=(8, 6))

# Plot customers: Loyal (blue) vs Churned (orange)
plt.scatter(X_test['Transactions'][y_test == 0], X_test['Support_Calls'][y_test == 0], 
            color='#3498db', label='Loyal (0)', edgecolors='k', alpha=0.8)
plt.scatter(X_test['Transactions'][y_test == 1], X_test['Support_Calls'][y_test == 1], 
            color='#e67e22', label='Churned (1)', edgecolors='k', alpha=0.8)

plt.title('SVM Classification of Customer Churn', fontsize=14, fontweight='bold')
plt.xlabel('Monthly Transactions')
plt.ylabel('Support Calls')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

