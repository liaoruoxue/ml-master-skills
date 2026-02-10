"""
Implementation 3.1: Random Forest
- Use best features from Direction 2
- Compare with Logistic Regression
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
test_labels = pd.read_csv('data/test_labels.csv')

def preprocess_best(df, train_df=None):
    """Best features from Direction 2."""
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
    fare = df['Fare'].fillna(train_df['Fare'].median())
    features['Fare'] = fare

    # FamilySize & IsAlone
    family_size = df['SibSp'] + df['Parch'] + 1
    features['FamilySize'] = family_size
    features['IsAlone'] = (family_size == 1).astype(int)

    # FareBin
    fare_binned = pd.cut(fare, bins=[-1, 7.91, 14.45, 31, 1000], labels=[0, 1, 2, 3])
    features['FareBin'] = fare_binned.fillna(1).astype(int)

    return features

# Prepare data
X_train = preprocess_best(train)
y_train = train['Survived']
X_test = preprocess_best(test, train)
y_test = test_labels['Survived']

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Test evaluation
model.fit(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

print(f"\nvs Best LR (81.46%): {'+' if test_acc > 0.8146 else ''}{(test_acc - 0.8146)*100:.2f}%")

# Feature importance
print("\nFeature Importance:")
for name, imp in sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.4f}")
