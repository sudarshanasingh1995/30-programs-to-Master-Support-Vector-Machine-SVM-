"""
svm_20_small_dataset_boston.py
-------------------------------
Concept: SVM on small housing features.
Algorithm: Support Vector Classifier (SVC) with a Linear Kernel.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Simulates a small housing dataset (30 records) containing House Size (sqft) and Bedrooms.
2. Formulates a classification goal (Expensive vs. Affordable).
3. Standardizes data.
4. Fits a linear SVM.
5. Prints predicted housing categories.
6. Displays a scatter plot mapping the housing separation boundary.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


print("--- Step 1: Generating Small Housing Dataset (30 samples) ---")
np.random.seed(42)

# Size in sqft (1000 to 4000)
size = np.random.uniform(1000, 4000, size=30)
# Number of bedrooms (2 to 6)
bedrooms = np.random.uniform(2, 6, size=30)

df = pd.DataFrame({
    'Size_sqft': size,
    'Bedrooms': bedrooms
})

# Pricing rule: Expensive (1) if size * 0.15 + bedrooms * 100 > 600
df['Expensive'] = ((df['Size_sqft'] * 0.15 + df['Bedrooms'] * 100) > 600).astype(int)

print("Housing Dataset (30 samples):")
print(df.head(10))
print(f"\nTarget distribution (0 = Affordable, 1 = Expensive):\n{df['Expensive'].value_counts()}")

print("\n--- Step 2: Normalizing and Splitting ---")
X = df[['Size_sqft', 'Bedrooms']]
y = df['Expensive']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Step 3: Training Linear SVM ---")
model = SVC(kernel='linear', C=1.0)
model.fit(X_scaled, y)
print("Model fit complete.")

print("\n--- Step 4: Checking Accuracy ---")
y_pred = model.predict(X_scaled)
acc = accuracy_score(y, y_pred)
print(f"Training accuracy on small dataset: {acc * 100:.1f}%")

print("\n--- Step 5: Visualizing Small Dataset Boundary ---")
plt.figure(figsize=(8, 6))

# Plot affordable (blue) vs expensive (orange)
plt.scatter(df['Size_sqft'][y == 0], df['Bedrooms'][y == 0], 
            color='#3498db', s=80, label='Affordable (0)', edgecolors='k')
plt.scatter(df['Size_sqft'][y == 1], df['Bedrooms'][y == 1], 
            color='#e67e22', s=80, label='Expensive (1)', edgecolors='k')

plt.title('SVM Pricing Classification (Small Dataset)', fontsize=12, fontweight='bold')
plt.xlabel('House Size (sqft)')
plt.ylabel('Number of Bedrooms')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

