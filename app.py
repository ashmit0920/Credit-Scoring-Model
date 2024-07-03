import streamlit as st
import numpy as np
import joblib
import pandas as pd
import altair as alt
import time

model = joblib.load("./exports/credit_scoring.pkl")
feature_sequence = joblib.load("./exports/feature_sequence.pkl")

data = pd.read_csv("./dataset2/CreditWorthiness.csv")

# Charts for Home tab
credit_score_counts = data['creditScore'].value_counts().reset_index()
credit_score_counts.columns = ['creditScore', 'Count']
cred_chart = alt.Chart(credit_score_counts).mark_bar().encode(
    x='creditScore',
    y='Count'
).properties(
    width=400,
)

age_counts = data['age'].value_counts().reset_index()
age_counts.columns = ['age', 'Count']
age_chart = alt.Chart(age_counts).mark_line().encode(
    x='age',
    y='Count'
)

camt_counts = data['Camt'].value_counts().reset_index()
camt_counts.columns = ['Credit Amount', 'Count']
camt_chart = alt.Chart(camt_counts).mark_bar().encode(
    x = alt.X('Credit Amount', bin = alt.BinParams(maxbins=30)),
    y='Count'
)

cdur_counts = data['Cdur'].value_counts().reset_index()
cdur_counts.columns = ['Credit Duration', 'Count']
cdur_chart = alt.Chart(cdur_counts).mark_bar().encode(
    x = alt.X('Credit Duration', bin = alt.BinParams(maxbins=20)),
    y = 'Count'
)

cpur_counts = data['Cpur'].value_counts().reset_index()
cpur_counts.columns = ['Credit Purpose', 'Count']
cpur_chart = alt.Chart(cpur_counts).mark_bar().encode(
    x = 'Credit Purpose',
    y = 'Count'
)

jobtype_counts = data['JobType'].value_counts().reset_index()
jobtype_counts.columns = ['Job Type', 'Count']
jobtype_chart = alt.Chart(jobtype_counts).mark_bar().encode(
    x = 'Job Type',
    y = 'Count'
)

footer = """
    <style>
    footer {
        visibility: hidden;
    }
    .main-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        border-top: 1px solid #e0e0e0
    }
    a {
        text-decoration: None;
    }
    </style>
    <div class="main-footer">
        <p>Credit Scoring App © 2024 | Built by Ashmit 👨‍💻 | For more projects, check out my <a href="https://github.com/ashmit0920">GitHub</a></p>
    </div>
"""

st.set_page_config(page_title='Credit Worthiness')

tab1, tab2, tab3 = st.tabs(["Home", "Predict", "About"])

# Streamlit app
with tab1:
    st.title(':green[Credit Scoring] Application')
    st.subheader("Assess creditworthiness of Loan applicants")
    st.write("Below are some :blue[visualizations] and :blue[data insights] based on the data that was used to train this model. To predict whether an applicant is creditworthy or not, head to the 'Predict' tab from the menu above.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Credit Score Distribution")
        st.altair_chart(cred_chart, use_container_width=True)

    with col2:
        st.markdown("#### Age Distribution")
        st.altair_chart(age_chart, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Credit Amount Distribution")
        st.altair_chart(camt_chart, use_container_width=True)

    with col4:
        st.markdown("#### Credit Duration Distribution")
        st.altair_chart(cdur_chart, use_container_width=True)
    
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("#### Credit Purpose Distribution")
        st.altair_chart(cpur_chart, use_container_width=True)
    
    with col6:
        st.markdown("#### Job Type Distribution")
        st.altair_chart(jobtype_chart, use_container_width=True)

    st.markdown(footer, unsafe_allow_html=True)
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
    'Cdur': 'Credit duration (in months)',
    'Camt': 'Loan amount',
    'InRate': 'Installment Rate',
    'NumCred': 'Number of existing credits',
}

# Displaying input options for categorical columns
with tab2:
    st.title(':green[Credit Scoring] Application')
    st.subheader("Assess creditworthiness of Loan applicants")
    st.write(f":orange[Note:] The predictions are made by a model trained on a relatively small dataset, which means it can not guarantee a 100% legit and real-world accurate prediction.")

    user_inputs = {}
    for col in features:
        if col in categorical:
            unique_values = input_options[col]
            user_inputs[col] = st.selectbox(f'Select {feature_names[col]}', options=unique_values)

        else:
            # numerical columns with number input
            if col in ['NumCred', 'InRate']:
                user_inputs[col] = st.number_input(f'Enter {feature_names[col]}', min_value=1, max_value=4, value=1)
            else:
                user_inputs[col] = st.number_input(f'Enter {feature_names[col]}')

    # Preparing input data for prediction
    user_data = pd.DataFrame([user_inputs])

    # Handling categorical columns with pd.get_dummies
    user_data = pd.get_dummies(user_data, columns=categorical)

    # Ensuring columns match the model's expectations
    missing_cols = set(feature_sequence) - set(user_data.columns)
    for col in missing_cols:
        user_data[col] = 0  # Set missing dummy columns to 0 if not provided by the user

    # Reordering the columns to match the order used during model training
    user_data = user_data[feature_sequence]

    if st.button('Predict Creditworthiness'):
        prediction = model.predict(user_data)
        
        with st.spinner("Predicting..."):
            time.sleep(2)
            if prediction == 1:
                st.success('The applicant is predicted to be creditworthy.')
            else:
                st.error('The applicant is predicted to default on the loan.')

with tab3:
    st.title(':green[Credit Scoring] Application')
    st.subheader("Assess creditworthiness of Loan applicants")
    st.markdown("Welcome to the Credit Scoring App! This application leverages machine learning to predict the creditworthiness of loan applicants. By analyzing 14 factors such as age, income, loan amount, employment status, credit history etc., the app utilizes a :blue[**Hyperparameter-tuned Gradient Boosting Classifier**] to provide accurate credit scores. The aim is to help financial institutions and loan officers make informed decisions quickly and efficiently. This tool showcases the power of predictive modeling in the finance sector, offering insights and transparency into the credit evaluation process. I hope you find this app useful and insightful :)")
    st.markdown("#### :orange[Disclaimer]")
    st.markdown(f"The ML model is trained on a relatively small dataset and is intended for demonstration purposes only, which means it can not guarantee a 100% legit and real-world accurate prediction. The developer does not assume any responsibility for the accuracy or reliability of predictions made using this app in real-world scenarios. Use at your own risk. The dataset used for training was sourced from Kaggle, and can be found [here](https://www.kaggle.com/datasets/bbjadeja/predicting-creditworthiness).")

    st.markdown(footer, unsafe_allow_html=True)