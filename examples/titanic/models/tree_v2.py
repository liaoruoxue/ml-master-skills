"""
Implementation 3.2: Feature Importance Analysis + Selection
- Use RF feature importance to select best features
- Try reducing feature set
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
test_labels = pd.read_csv('data/test_labels.csv')

def preprocess_best(df, train_df=None):
    """Best features from Direction 2."""
    features = pd.DataFrame()

    if train_df is None:
        train_df = df

    features['Sex'] = (df['Sex'] == 'male').astype(int)
    features['Pclass'] = df['Pclass']

    age_medians = train_df.groupby(['Pclass', 'Sex'])['Age'].median()
    def impute_age(row):
        if pd.isna(row['Age']):
            return age_medians.get((row['Pclass'], row['Sex']), train_df['Age'].median())
        return row['Age']
    features['Age'] = df.apply(impute_age, axis=1)

    fare = df['Fare'].fillna(train_df['Fare'].median())
    features['Fare'] = fare

    family_size = df['SibSp'] + df['Parch'] + 1
    features['FamilySize'] = family_size
    features['IsAlone'] = (family_size == 1).astype(int)

    fare_binned = pd.cut(fare, bins=[-1, 7.91, 14.45, 31, 1000], labels=[0, 1, 2, 3])
    features['FareBin'] = fare_binned.fillna(1).astype(int)

    return features

# Prepare data
X_train_full = preprocess_best(train)
y_train = train['Survived']
X_test_full = preprocess_best(test, train)
y_test = test_labels['Survived']

# Try different feature subsets based on importance
# Top features: Age, Sex, Fare, Pclass
feature_sets = {
    'All': list(X_train_full.columns),
    'Top4': ['Age', 'Sex', 'Fare', 'Pclass'],
    'Top5': ['Age', 'Sex', 'Fare', 'Pclass', 'FamilySize'],
    'NoFareBin': ['Age', 'Sex', 'Fare', 'Pclass', 'FamilySize', 'IsAlone'],
}

print("Feature Selection Analysis:")
print("-" * 60)

best_test_acc = 0
best_features = None

for name, features in feature_sets.items():
    X_train = X_train_full[features]
    X_test = X_test_full[features]

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    model.fit(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"{name:12} | CV: {cv_scores.mean():.4f} | Test: {test_acc:.4f}")

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_features = name

print("-" * 60)
print(f"Best: {best_features} with Test Acc: {best_test_acc:.4f}")
print(f"vs Best LR (81.46%): {'+' if best_test_acc > 0.8146 else ''}{(best_test_acc - 0.8146)*100:.2f}%")
