"""
Implementation 2.2: FamilySize Feature
- FamilySize = SibSp + Parch + 1
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
test_labels = pd.read_csv('data/test_labels.csv')

def preprocess_fe_v2(df, train_df=None, scaler=None, fit_scaler=True):
    """With FamilySize feature."""
    features = pd.DataFrame()

    if train_df is None:
        train_df = df

    # Sex
    features['Sex'] = (df['Sex'] == 'male').astype(int)

    # Pclass
    features['Pclass'] = df['Pclass']

    # Age
    age_medians = train_df.groupby(['Pclass', 'Sex'])['Age'].median()
    def impute_age(row):
        if pd.isna(row['Age']):
            return age_medians.get((row['Pclass'], row['Sex']), train_df['Age'].median())
        return row['Age']
    features['Age'] = df.apply(impute_age, axis=1)

    # Fare
    features['Fare'] = df['Fare'].fillna(train_df['Fare'].median())

    # NEW: FamilySize
    features['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Scaling
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    else:
        features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)

    return features_scaled, scaler

# Prepare data
X_train, scaler = preprocess_fe_v2(train)
y_train = train['Survived']
X_test, _ = preprocess_fe_v2(test, train, scaler, fit_scaler=False)
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

print(f"\nvs Best (78.09%): {'+' if test_acc > 0.7809 else ''}{(test_acc - 0.7809)*100:.2f}%")

# Analyze FamilySize impact
print("\nFamilySize Coefficients:")
for name, coef in zip(X_train.columns, model.coef_[0]):
    print(f"  {name}: {coef:.4f}")
