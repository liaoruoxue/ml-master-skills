"""
Implementation 2.1: Extract Title from Name
- Extract Mr, Mrs, Miss, Master, etc.
- Group rare titles
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

def extract_title(name):
    """Extract title from name."""
    import re
    title_search = re.search(r' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def preprocess_fe_v1(df, train_df=None, scaler=None, fit_scaler=True):
    """With Title feature."""
    features = pd.DataFrame()

    # Sex
    features['Sex'] = (df['Sex'] == 'male').astype(int)

    # Pclass
    features['Pclass'] = df['Pclass']

    # Age: impute by median
    if train_df is None:
        train_df = df
    age_medians = train_df.groupby(['Pclass', 'Sex'])['Age'].median()
    def impute_age(row):
        if pd.isna(row['Age']):
            return age_medians.get((row['Pclass'], row['Sex']), train_df['Age'].median())
        return row['Age']
    features['Age'] = df.apply(impute_age, axis=1)

    # Fare
    features['Fare'] = df['Fare'].fillna(train_df['Fare'].median())

    # NEW: Title extraction
    df = df.copy()
    df['Title'] = df['Name'].apply(extract_title)

    # Group rare titles
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Officer',
        'Rev': 'Officer',
        'Col': 'Officer',
        'Major': 'Officer',
        'Mlle': 'Miss',
        'Countess': 'Royalty',
        'Ms': 'Miss',
        'Lady': 'Royalty',
        'Jonkheer': 'Royalty',
        'Don': 'Royalty',
        'Dona': 'Royalty',
        'Mme': 'Mrs',
        'Capt': 'Officer',
        'Sir': 'Royalty'
    }
    df['Title'] = df['Title'].map(lambda x: title_mapping.get(x, 'Other'))

    # One-hot encode Title
    for title in ['Mr', 'Miss', 'Mrs', 'Master', 'Officer', 'Royalty']:
        features[f'Title_{title}'] = (df['Title'] == title).astype(int)

    # Scaling
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    else:
        features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)

    return features_scaled, scaler

# Prepare data
X_train, scaler = preprocess_fe_v1(train)
y_train = train['Survived']
X_test, _ = preprocess_fe_v1(test, train, scaler, fit_scaler=False)
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

print(f"\nFeatures: {list(X_train.columns)}")
print(f"vs Best (78.09%): {'+' if test_acc > 0.7809 else ''}{(test_acc - 0.7809)*100:.2f}%")
