# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Generate example dataset (replace with your actual dataset)
np.random.seed(0)
n_samples = 1000

# Example features: usage behavior and customer demographics
usage_minutes = np.random.normal(500, 150, n_samples)
age = np.random.randint(18, 70, n_samples)
income = np.random.normal(50000, 10000, n_samples)
is_subscriber = np.random.randint(0, 2, n_samples)  # 0 = non-subscriber, 1 = subscriber

# Example target: churn (0 = retained, 1 = churned)
churn_rate = 0.3
churned = np.random.binomial(1, churn_rate, n_samples)

# Create DataFrame
data = pd.DataFrame({
    'usage_minutes': usage_minutes,
    'age': age,
    'income': income,
    'is_subscriber': is_subscriber,
    'churned': churned
})

# Separate features (X) and target (y)
X = data.drop('churned', axis=1)
y = data['churned']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Scale numerical features and encode categorical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Train and evaluate models
for clf_name, clf in classifiers.items():
    print(f"Training {clf_name}...")
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    
    # Evaluate model
    print(f"\n{clf_name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("="*60)
