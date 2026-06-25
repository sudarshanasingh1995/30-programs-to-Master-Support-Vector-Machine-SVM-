"""
svm_08_linear_kernel_text_spam.py
---------------------------------
Concept: Text Classification using Linear Kernel SVM.
Algorithm: Support Vector Classifier (SVC) with a Linear Kernel.
Aesthetics: Modern, beginner-friendly script with detailed comments.

This program:
1. Defines a small, representative collection of text messages (Spam vs. Ham).
2. Converts raw text to numerical features using a Bag-of-Words Vectorizer.
3. Fits a linear SVM classifier to learn separating keyword boundaries.
4. Predicts classes on new, unseen test phrases.
5. Displays a bar chart illustrating word frequency importances (coefficients).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


print("--- Step 1: Defining Text Dataset ---")
# A simple dataset of messages
corpus = [
    "Win a free cash prize now! Click here to claim your reward.",
    "URGENT: Your account has been suspended. Reply to reactivate.",
    "Get cheap loans instantly. No credit check required.",
    "Earn extra money working from home in just three hours.",
    "Congratulations! You won a gift card. Call now.",
    "Hey, what time are we meeting for dinner tonight?",
    "Can you send me the report by this afternoon?",
    "Thanks for the coffee, I really appreciate it.",
    "Let me know if you want to hang out this weekend.",
    "Hello, I will be late for the meeting today."
]

# Labels: 1 = Spam, 0 = Ham (Not Spam)
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

df = pd.DataFrame({
    'Message': corpus,
    'Spam': labels
})

print(df)

print("\n--- Step 2: Extracting Text Features (Bag-of-Words) ---")
# Convert messages to lowercase and tokenize
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Message']).toarray()
y = df['Spam']

words = vectorizer.get_feature_names_out()
print(f"Vocabulary size (unique words): {len(words)}")
print("Sample extracted words:")
print(words[:15])

print("\n--- Step 3: Training Linear SVM ---")
# Linear kernels are excellent for text classification due to high dimensionality.
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)
print("Linear text classifier trained successfully.")

print("\n--- Step 4: Predicting on Unseen Test Messages ---")
test_messages = [
    "Hey dinner is ready, come over",
    "URGENT: Click here to claim your free cash prize!"
]

X_test = vectorizer.transform(test_messages).toarray()
predictions = model.predict(X_test)

for msg, pred in zip(test_messages, predictions):
    label_str = "SPAM" if pred == 1 else "HAM (Not Spam)"
    print(f"Message: '{msg}' -> Predicted: {label_str}")

print("\n--- Step 5: Visualizing Word Coefficients ---")
# A linear kernel's coefficients represent the relative importance of words.
coef = model.coef_[0]
importance_df = pd.DataFrame({
    'Word': words,
    'Coefficient': coef
}).sort_values(by='Coefficient', ascending=False)

# Get top 5 spam words and top 5 ham words
top_words = pd.concat([importance_df.head(5), importance_df.tail(5)])

plt.figure(figsize=(10, 5))
colors = ['#e74c3c' if val > 0 else '#3498db' for val in top_words['Coefficient']]
plt.barh(top_words['Word'], top_words['Coefficient'], color=colors, edgecolor='k')
plt.axvline(0, color='black', linestyle='--', alpha=0.7)
plt.title('Top Keyword Influences (Positive = Spam, Negative = Ham)', fontsize=14, fontweight='bold')
plt.xlabel('SVM Feature Importance (Coefficient)')
plt.grid(True, alpha=0.3)

plt.show()
print("Program completed successfully!")

