"""
Implementation 1.2: Improved Missing Value Handling
- Age: median imputation by Pclass and Sex
- Add Fare feature with median imputation
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
test_labels = pd.read_csv('data/test_labels.csv')

def preprocess_v2(df, train_df=None):
    """Improved preprocessing with better imputation."""
    features = pd.DataFrame()

    # Sex: encode as 0/1
    features['Sex'] = (df['Sex'] == 'male').astype(int)

    # Pclass
    features['Pclass'] = df['Pclass']

    # Age: impute by median of Pclass+Sex group
    if train_df is None:
        train_df = df
    age_medians = train_df.groupby(['Pclass', 'Sex'])['Age'].median()

    def impute_age(row):
        if pd.isna(row['Age']):
            return age_medians.get((row['Pclass'], row['Sex']), train_df['Age'].median())
        return row['Age']

    features['Age'] = df.apply(impute_age, axis=1)

    # Fare: median imputation
    features['Fare'] = df['Fare'].fillna(train_df['Fare'].median())

    return features

# Prepare data
X_train = preprocess_v2(train)
y_train = train['Survived']
X_test = preprocess_v2(test, train)
y_test = test_labels['Survived']

# Train model
model = LogisticRegression(random_state=42, max_iter=1000)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Test evaluation
model.fit(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

print("\nFeature Coefficients:")
for name, coef in zip(X_train.columns, model.coef_[0]):
    print(f"  {name}: {coef:.4f}")
