# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset (replace 'creditcard.csv' with your dataset path)
df = pd.read_csv('creditcard.csv')

# Explore the dataset
print(df.head())
print(df.info())

# Check class distribution
print(df['Class'].value_counts())  # 0: legitimate, 1: fraudulent

# Separate features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
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

# Example of predicting on a single transaction
# Assuming new_transaction is a pandas DataFrame with the same structure as X (excluding 'Class')
new_transaction = pd.DataFrame({
    'Time': [12345],
    'V1': [-1.5],
    'V2': [2.5],
    'V3': [-2.0],
    'V4': [1.5],
    'V5': [-2.5],
    'V6': [3.0],
    'V7': [-0.5],
    'V8': [0.2],
    'V9': [-1.2],
    'V10': [0.5],
    'V11': [-1.5],
    'V12': [0.8],
    'V13': [-0.5],
    'V14': [-1.2],
    'V15': [0.3],
    'V16': [-1.5],
    'V17': [0.6],
    'V18': [-1.0],
    'V19': [0.8],
    'V20': [-0.2],
    'V21': [0.1],
    'V22': [-0.3],
    'V23': [0.2],
    'V24': [-0.5],
    'V25': [0.3],
    'V26': [-0.7],
    'V27': [0.1],
    'V28': [-0.3],
    'Amount': [256.5]
})

# Scale the new transaction data
new_transaction_scaled = scaler.transform(new_transaction)

# Predict using Random Forest (you can choose any trained model here)
predicted_class = classifiers['Random Forest'].predict(new_transaction_scaled)
print("\nPredicted class for the new transaction:", predicted_class[0])
