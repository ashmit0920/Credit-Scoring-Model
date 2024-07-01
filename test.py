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

# Resampling data as good:bad is 700:300
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Train a Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

model = GradientBoostingClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Set up Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluation
from sklearn.metrics import roc_auc_score, classification_report

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best ROC-AUC Score from Grid Search: {best_score}")

# Evaluate on test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC Score on Test Data: {roc_auc}")

class_report = classification_report(y_test, y_pred)

print("Classification Report:")
print(class_report)