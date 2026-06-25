"""
svm_27_accuracy_roc_auc.py
--------------------------
Concept: ROC Curve & AUC Score Evaluation.
Algorithm: Support Vector Classifier (SVC) with Probability Outputs.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Generates a noisy synthetic binary classification dataset.
2. Fits an RBF SVM (with `probability=True` enabled).
3. Computes class prediction probabilities for the test set.
4. Calculates False Positive Rates, True Positive Rates, and the AUC metric.
5. Prints the AUC evaluation score.
6. Saves the plotted ROC curve with diagonal baseline references.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score


print("--- Step 1: Generating Noisy Dataset ---")
# Generate a slightly noisy dataset so the ROC curve has an interesting curved shape rather than a perfect corner
X, y = make_classification(n_samples=250, n_features=5, n_informative=3, flip_y=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data standardized.")

print("\n--- Step 2: Training SVM with Probability Outputs ---")
# To plot an ROC curve, we need prediction probabilities.
# We MUST set probability=True on initialization of SVC.
model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
model.fit(X_train_scaled, y_train)
print("RBF SVM trained with probability enabled.")

print("\n--- Step 3: Computing Probabilities and ROC Curve ---")
# Predict probabilities: Output shape is (n_samples, 2). Column 1 is probability of positive class.
y_probs = model.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calculate Area Under the Curve (AUC)
auc_score = roc_auc_score(y_test, y_probs)
print(f"ROC AUC Score: {auc_score:.4f} ({auc_score * 100:.1f}% perfect classifier baseline)")

print("\n--- Step 4: Plotting and Saving the ROC Curve ---")
plt.figure(figsize=(8, 7))

# Plot the ROC Curve
plt.plot(fpr, tpr, color='#e67e22', linewidth=3, label=f'SVM ROC Curve (AUC = {auc_score:.3f})')

# Plot the diagonal line representing random guessing (AUC = 0.5)
plt.plot([0, 1], [0, 1], color='#34495e', linestyle='--', linewidth=1.5, label='Random Guessing (AUC = 0.500)')

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

