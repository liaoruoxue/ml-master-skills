"""
Implementation 4.2: Hyperparameter Tuning
- GridSearchCV for GradientBoosting
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
test_labels = pd.read_csv('data/test_labels.csv')

def preprocess_best(df, train_df=None):
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
X_train = preprocess_best(train)
y_train = train['Survived']
X_test = preprocess_best(test, train)
y_test = test_labels['Survived']

# Grid search for GradientBoosting
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

gb = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
print(f"Best Parameters: {grid_search.best_params_}")

# Test with best model
best_model = grid_search.best_estimator_
test_acc = best_model.score(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

print(f"\nvs Best Ensemble (83.15%): {'+' if test_acc > 0.8315 else ''}{(test_acc - 0.8315)*100:.2f}%")
