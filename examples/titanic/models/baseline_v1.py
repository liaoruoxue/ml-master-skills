"""
Implementation 1.1: Logistic Regression Baseline
Features: Sex, Pclass, Age (with simple imputation)
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
test_labels = pd.read_csv('data/test_labels.csv')

# Basic preprocessing
def preprocess_v1(df):
    """Minimal preprocessing for baseline."""
    features = pd.DataFrame()

    # Sex: encode as 0/1
    features['Sex'] = (df['Sex'] == 'male').astype(int)

    # Pclass: already numeric
    features['Pclass'] = df['Pclass']

    # Age: simple mean imputation
    features['Age'] = df['Age'].fillna(df['Age'].mean())

    return features

# Prepare data
X_train = preprocess_v1(train)
y_train = train['Survived']
X_test = preprocess_v1(test)
y_test = test_labels['Survived']

# Train model
model = LogisticRegression(random_state=42, max_iter=1000)

# Cross-validation on training set
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Final evaluation on test set
model.fit(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Feature importance
print("\nFeature Coefficients:")
for name, coef in zip(X_train.columns, model.coef_[0]):
    print(f"  {name}: {coef:.4f}")
