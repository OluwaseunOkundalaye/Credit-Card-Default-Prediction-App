import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("xgboost_credit_model.pkl")   # trained XGBoost model
scaler = joblib.load("scaler.pkl")     # trained StandardScaler

st.set_page_config(page_title="Credit Default Predictor", layout="centered")
st.title("💳 Credit Card Default Prediction App")
st.write("Predict the probability of a client defaulting next month.")

# -------------------- USER INPUT --------------------
st.header("Client Information")

col1, col2 = st.columns(2)
with col1:
    LIMIT_BAL = st.number_input("Credit Limit (NT dollars)", min_value=0, value=20000)
    SEX = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x==1 else "Female")
    EDUCATION = st.selectbox("Education", options=[1,2,3,4],
                             format_func=lambda x: {1:"Grad School",2:"University",3:"High School",4:"Others"}[x])
    MARRIAGE = st.selectbox("Marital Status", options=[1,2,3],
                            format_func=lambda x: {1:"Married",2:"Single",3:"Others"}[x])
    AGE = st.number_input("Age", min_value=18, max_value=100, value=30)

with col2:
    PAY_0 = st.number_input("Repayment Status Sep 2005 (-2 to 8)", min_value=-2, max_value=8, value=0)
    PAY_2 = st.number_input("Repayment Status Aug 2005 (-2 to 8)", min_value=-2, max_value=8, value=0)
    PAY_3 = st.number_input("Repayment Status Jul 2005 (-2 to 8)", min_value=-2, max_value=8, value=0)
    PAY_4 = st.number_input("Repayment Status Jun 2005 (-2 to 8)", min_value=-2, max_value=8, value=0)
    PAY_5 = st.number_input("Repayment Status May 2005 (-2 to 8)", min_value=-2, max_value=8, value=0)
    PAY_6 = st.number_input("Repayment Status Apr 2005 (-2 to 8)", min_value=-2, max_value=8, value=0)

st.header("Billing Information")
BILL_AMT1 = st.number_input("Bill Amount Sep 2005", min_value=0, value=0)
BILL_AMT2 = st.number_input("Bill Amount Aug 2005", min_value=0, value=0)
BILL_AMT3 = st.number_input("Bill Amount Jul 2005", min_value=0, value=0)
BILL_AMT4 = st.number_input("Bill Amount Jun 2005", min_value=0, value=0)
BILL_AMT5 = st.number_input("Bill Amount May 2005", min_value=0, value=0)
BILL_AMT6 = st.number_input("Bill Amount Apr 2005", min_value=0, value=0)

st.header("Previous Payments")
PAY_AMT1 = st.number_input("Previous Payment Sep 2005", min_value=0, value=0)
PAY_AMT2 = st.number_input("Previous Payment Aug 2005", min_value=0, value=0)
PAY_AMT3 = st.number_input("Previous Payment Jul 2005", min_value=0, value=0)
PAY_AMT4 = st.number_input("Previous Payment Jun 2005", min_value=0, value=0)
PAY_AMT5 = st.number_input("Previous Payment May 2005", min_value=0, value=0)
PAY_AMT6 = st.number_input("Previous Payment Apr 2005", min_value=0, value=0)

# -------------------- PREDICTION --------------------
if st.button("Predict Default"):
    
    # Feature Engineering
    TOTAL_BILL = BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6
    TOTAL_PAY = PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6
    UTILIZATION = TOTAL_BILL / (LIMIT_BAL + 1)  # +1 to avoid div by zero
    PAYMENT_RATIO = TOTAL_PAY / (TOTAL_BILL + 1)
    
    # Create DataFrame
    input_df = pd.DataFrame({
        'LIMIT_BAL':[LIMIT_BAL], 'AGE':[AGE],
        'PAY_0':[PAY_0], 'PAY_2':[PAY_2], 'PAY_3':[PAY_3], 'PAY_4':[PAY_4],
        'PAY_5':[PAY_5], 'PAY_6':[PAY_6],
        'BILL_AMT1':[BILL_AMT1], 'BILL_AMT2':[BILL_AMT2], 'BILL_AMT3':[BILL_AMT3],
        'BILL_AMT4':[BILL_AMT4], 'BILL_AMT5':[BILL_AMT5], 'BILL_AMT6':[BILL_AMT6],
        'PAY_AMT1':[PAY_AMT1], 'PAY_AMT2':[PAY_AMT2], 'PAY_AMT3':[PAY_AMT3],
        'PAY_AMT4':[PAY_AMT4], 'PAY_AMT5':[PAY_AMT5], 'PAY_AMT6':[PAY_AMT6],
        'TOTAL_BILL':[TOTAL_BILL], 'TOTAL_PAY':[TOTAL_PAY],
        'UTILIZATION':[UTILIZATION], 'PAYMENT_RATIO':[PAYMENT_RATIO],
        'SEX':[SEX], 'EDUCATION':[EDUCATION], 'MARRIAGE':[MARRIAGE]
    })

    # One-hot encode categorical variables
    input_df = pd.get_dummies(input_df, columns=['SEX','EDUCATION','MARRIAGE'])
    
    # Align input with training features
    for col in model.get_booster().feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.get_booster().feature_names]

    # Scale numeric columns
    numeric_cols = ['LIMIT_BAL','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
                    'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
                    'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6',
                    'TOTAL_BILL','TOTAL_PAY','UTILIZATION','PAYMENT_RATIO']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Prediction
    prediction = model.predict(input_df)[0]
    prediction_prob = model.predict_proba(input_df)[0][1]

    # Display
    if prediction == 1:
        st.error(f"⚠ The model predicts: DEFAULT (probability = {prediction_prob:.2f})")
    else:
        st.success(f"✅ The model predicts: NO DEFAULT (probability = {prediction_prob:.2f})")
