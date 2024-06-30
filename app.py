import streamlit as st
import numpy as np
import pickle
import joblib

model = joblib.load("credit_scoring_using_gradient_boosting.pkl")

# Streamlit app
st.set_page_config(page_title='Credit Worthiness')
st.title('Credit Scoring Application')

# Input features
age = st.number_input('Age', min_value=18, max_value=70, value=30)
income = st.number_input('Income', min_value=20000, max_value=100000, value=50000)
loan_amount = st.number_input('Loan Amount', min_value=1000, max_value=50000, value=10000)
loan_duration = st.number_input('Loan Duration (months)', min_value=1, max_value=30, value=12)

# Predict button
if st.button('Predict Creditworthiness'):
    features = np.array([[age, income, loan_amount, loan_duration]])
    prediction = model.predict(features)
    
    if prediction == 0:
        st.success('The applicant is predicted to be creditworthy.')
    else:
        st.error('The applicant is predicted to default on the loan.')