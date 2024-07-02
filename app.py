import streamlit as st
import numpy as np
import pickle
import joblib
import pandas as pd

model = joblib.load("credit_scoring.pkl")

# Streamlit app
st.set_page_config(page_title='Credit Worthiness')
st.title('Credit Scoring Application')

# Input features
features = ['age', 'Cdur', 'Camt', 'NumCred', 'Cbal', 'Chist', 'Cpur', 'Sbal', 'Edur', 'InRate', 'MSG', 'Oparties', 'JobType', 'Rdur']
categorical = ['Cbal', 'Chist', 'Cpur', 'Sbal', 'Edur', 'MSG', 'Oparties', 'JobType', 'Rdur']

input_options = {
    "Cbal": ["0 <= Rs. < 2000", "no checking account", " Rs. < 0", "Rs. >=2000"], 
    "Chist": ["all settled till now", "dues not paid earlier", "none taken/all settled", "all settled"], 
    "Cpur": ["Business", "electronics", "renovation", "second hand vehicle", "education", "new vehicle", "miscellaneous", "furniture", "retaining", "domestic needs"], 
    "Sbal": ["Rs. < 1000", "no savings account", "Rs. >= 10,000", "5000 <= Rs. < 10,000", "1000 <= Rs. < 5,000"], 
    "Edur": ["1 to 4 years", "more than 7 years", "less than 1 year", "4 to 7 years", "not employed"], 
    "MSG": ["married or widowed male", "single male", "divorced or separated or married female", "divorced or separated male"], 
    "Oparties": ["no one", "yes, guarantor", "yes, co-applicant"], 
    "Rdur": ["less than a year", "more than 3 years", "1 to 2 years", "2 to 3 years"], 
    "JobType": ["employee with official position", "employed either in management, self or in high position", "resident unskilled", "non resident either unemployed or  unskilled "]
}

feature_names = {
    'Cbal': 'Checking account balance',
    'Chist': 'Credit history',
    'Cpur': 'Purpose of credit',
    'Sbal': 'Savings account balance',
    'Edur': 'Employment duration',
    'MSG': 'Marital Status and Gender',
    'Oparties': 'Other parties',
    'Rdur': 'Residence duration',
    'JobType': 'Job type',
    'age': 'Age',
    'Cdur': 'Credit duration',
    'Camt': 'Loan amount',
    'InRate': 'Installment Rate',
    'NumCred': 'Number of existing credits',
}

# Displaying input options for categorical columns
user_inputs = {}
for col in features:
    if col in categorical:
        unique_values = input_options[col]
        user_inputs[col] = st.selectbox(f'Select {feature_names[col]}', options=unique_values)

    else:
        # Handle numerical columns with text input
        if col in ['NumCred', 'InRate']:
            user_inputs[col] = st.number_input(f'Enter {feature_names[col]}', min_value=1, max_value=4, value=1)
        else:
            user_inputs[col] = st.number_input(f'Enter {feature_names[col]}')

# Preparing input data for prediction
user_data = pd.DataFrame([user_inputs])

# Handling categorical columns with pd.get_dummies
user_data = pd.get_dummies(user_data, columns=categorical)

# Ensuring columns match the model's expectations
missing_cols = set(features) - set(user_data.columns)
for col in missing_cols:
    user_data[col] = 0  # Set missing dummy columns to 0 if not provided by the user

# Predict button
if st.button('Predict Creditworthiness'):
    prediction = model.predict(user_data)
    
    if prediction == 1:
        st.success('The applicant is predicted to be creditworthy.')
    else:
        st.error('The applicant is predicted to default on the loan.')