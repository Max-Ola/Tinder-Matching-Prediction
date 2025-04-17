import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate synthetic data
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'user1_age': np.random.randint(18, 45, n),
    'user2_age': np.random.randint(18, 45, n),
    'distance_km': np.random.randint(1, 50, n),
    'common_interests': np.random.randint(0, 10, n),
    'user1_attractiveness': np.random.randint(1, 11, n),
    'user2_attractiveness': np.random.randint(1, 11, n),
})

# Simulate matches
def simulate_match(row):
    if abs(row['user1_age'] - row['user2_age']) < 5 and row['distance_km'] < 15 and row['common_interests'] > 2 and abs(row['user1_attractiveness'] - row['user2_attractiveness']) <= 3:
        return 1
    return 0

data['match'] = data.apply(simulate_match, axis=1)

# Split features and target
X = data.drop('match', axis=1)
y = data['match']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))