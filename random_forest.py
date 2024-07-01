import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("./dataset2/CreditWorthiness.csv")

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

# Resampling data as good:bad is 700:300
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"ROC_AUC: {roc_auc_score(y_test, y_pred)}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
