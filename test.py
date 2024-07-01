import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("/content/CreditWorthiness.csv")

features = ['age', 'Cdur', 'Camt', 'NumCred', 'Cbal', 'Chist', 'Cpur', 'Sbal', 'Edur', 'MSG', 'Oparties', 'JobType', 'creditScore', 'Rdur']
data = data[features]

# Convert categorical features to numeric
data = pd.get_dummies(data, columns=['Cbal', 'Chist', 'Cpur', 'Sbal', 'Edur', 'MSG', 'Oparties', 'JobType', 'Rdur'])

# Split data into features and target
X = data.drop('creditScore', axis=1)
y = data['creditScore']

# Convert target variable to numerical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y) # good -> 1, bad -> 0

# Standardize numerical features if necessary
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[['age', 'Cdur', 'Camt', 'NumCred']] = scaler.fit_transform(X[['age', 'Cdur', 'Camt', 'NumCred']])

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import roc_auc_score, classification_report
y_pred = model.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"ROC-AUC Score: {roc_auc}")
print("Classification Report:")
print(class_report)