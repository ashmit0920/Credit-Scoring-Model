# Credit Scoring App 

## Overview

The [Credit Scoring App](https://credit-scoring-model.streamlit.app) is a machine learning-powered web application that assesses the creditworthiness of loan applicants. The app is built using Streamlit and leverages a Hyperparameter-tuned Gradient Boosting Classifier trained on a real-world dataset to predict whether an applicant's credit score is "good" or "bad". This project is intended for demonstration purposes and currently should not be used for actual credit assessments.

## Features

- User Interface: Interactive web interface for users to input applicant details and view infographics.
- Prediction: Real-time prediction of credit score (good or bad) based on user inputs.
- Visualization: Interactive charts and graphs to visualize data distributions and model predictions.

## Usage

1. Open the [app](https://credit-scoring-model.streamlit.app) in your web browser.
2. View interactive charts in the Home tab to analyze the data distribution.
3. Head to the Predict tab and enter the required details for the loan applicant.
4. Click on the "Predict" button to get the credit score prediction.

## Installation

To run this app locally, follow these steps:
1. Clone the repository: 
```
git clone https://github.com/ashmit0920/Credit-Scoring-Model
cd Credit-Scoring-Model
```

2. Create a virtual environment and activate it:
```
python3 -m venv venv
venv\Scripts\activate # On Linux, use 'source venv/bin/activate'
```

3. Install the dependencies:
```
pip install -r requirements.txt
```
4. Run the Streamlit app:
```
streamlit run app.py
```
5. Open your web browser and navigate to ```http://localhost:8501``` to view the app.

## Disclaimer

This application is trained on a relatively small dataset and is intended for demonstration purposes only. The developer does not assume any responsibility for the accuracy or reliability of predictions made using this app in real-world scenarios. Use at your own risk. The dataset used for training was sourced from Kaggle, and can be found [here](https://www.kaggle.com/datasets/bbjadeja/predicting-creditworthiness)."

## Future Enhancements

- More detailed input validation and error handling.
- Integrating more sophisticated model training and data visualization techniques.
- Exploring the possibility of integrating a LLM for enhanced feature extraction and prediction.
