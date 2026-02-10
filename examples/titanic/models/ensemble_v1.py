"""
Implementation 4.1: Voting Ensemble
- Combine LR + RF + GradientBoosting
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
test_labels = pd.read_csv('data/test_labels.csv')

def preprocess_best(df, train_df=None):
    """Best features."""
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

# Scale for LR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Individual models
lr = LogisticRegression(random_state=42, max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)

# Voting ensemble (soft voting)
ensemble = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
    voting='soft'
)

# Cross-validation
cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Ensemble CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Test evaluation
ensemble.fit(X_train_scaled, y_train)
test_acc = ensemble.score(X_test_scaled, y_test)
print(f"Ensemble Test Accuracy: {test_acc:.4f}")

print(f"\nvs Best GB (82.58%): {'+' if test_acc > 0.8258 else ''}{(test_acc - 0.8258)*100:.2f}%")

# Compare individual models
print("\nIndividual Model Comparison:")
for name, model in [('LR', lr), ('RF', rf), ('GB', gb)]:
    model.fit(X_train_scaled, y_train)
    acc = model.score(X_test_scaled, y_test)
    print(f"  {name}: {acc:.4f}")
