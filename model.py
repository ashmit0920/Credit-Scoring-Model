import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Generate synthetic dataset
np.random.seed(42)

num_samples = 1000
data = pd.DataFrame({
    'age': np.random.randint(18, 70, size=num_samples),
    'income': np.random.randint(20000, 100000, size=num_samples),
    'loan_amount': np.random.randint(1000, 50000, size=num_samples),
    'loan_duration': np.random.randint(1, 30, size=num_samples),
    'default': np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])
})

# Display first few rows of the dataset
data.head()

# Data Pre-processing
data.isnull().sum()

X = data.drop('default', axis=1)
y = data['default']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Development and Evaluation

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Feature Importance

feature_importances = model.feature_importances_
features = X.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Credit Scoring Model')
plt.gca().invert_yaxis()
plt.show()


# Save the model
joblib.dump(model, 'credit_scoring_model.pkl')