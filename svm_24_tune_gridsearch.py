"""
svm_24_tune_gridsearch.py
-------------------------
Concept: Automated Hyperparameter Tuning.
Algorithm: Support Vector Classifier (SVC) with GridSearchCV.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates a synthetic classification dataset.
2. Standardizes feature parameters.
3. Defines a parameter grid containing C values and gamma values.
4. Executes `GridSearchCV` using a 5-fold cross-validation strategy.
5. Prints the best parameters and best estimator score.
6. Displays a heatmap plot displaying cross-validation accuracy across parameter combinations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


print("--- Step 1: Generating Dataset ---")
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data standardized.")

print("\n--- Step 2: Setting up GridSearchCV ---")
# Define parameter search grid
param_grid = {
    'C': [0.1, 1.0, 10.0, 100.0],
    'gamma': [0.01, 0.1, 1.0, 10.0]
}

# Base model: RBF kernel SVM
base_model = SVC(kernel='rbf')

# Initialize GridSearchCV with 5-fold cross validation
grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

print("Running parameter grid search...")
grid_search.fit(X_train_scaled, y_train)
print("Grid search completed successfully.")

print("\n--- Step 3: Extracting Best Tuning Results ---")
print(f"Best parameter combination found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy score: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.1f}%)")

# Evaluate on holdout test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of best model on holdout test set: {test_accuracy * 100:.1f}%")

print("\n--- Step 4: Visualizing Grid Search Results ---")
# Reshape the grid search results to fit a heatmap structure
results = pd.DataFrame(grid_search.cv_results_)
scores_matrix = results.pivot(index='param_C', columns='param_gamma', values='mean_test_score').values

plt.figure(figsize=(8, 6))
# Plot matrix values as heatmap
im = plt.imshow(scores_matrix, cmap='viridis', origin='lower')
plt.colorbar(im, label='Mean CV Accuracy')

# Label ticks
c_labels = param_grid['C']
gamma_labels = param_grid['gamma']

plt.xticks(np.arange(len(gamma_labels)), gamma_labels)
plt.yticks(np.arange(len(c_labels)), c_labels)

plt.xlabel('Kernel Gamma Parameter')
plt.ylabel('Regularization C Parameter')
plt.title('GridSearchCV Accuracy Heatmap', fontsize=14, fontweight='bold')

# Annotate values inside the matrix grid
for i in range(len(c_labels)):
    for j in range(len(gamma_labels)):
        score = scores_matrix[i, j]
        plt.text(j, i, f"{score*100:.1f}%", ha="center", va="center", 
                 color="white" if score < 0.85 else "black", fontweight='bold')

plt.show()
print("Program completed successfully!")

