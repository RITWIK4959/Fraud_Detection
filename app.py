import os
import joblib
import numpy as np
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

model = joblib.load(os.path.join(MODEL_DIR, "xgb_fraud_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

st.set_page_config(page_title="Fraud Detection System", layout="centered")
st.title("Fraud Detection System")
st.write("Enter transaction details below to check if it's fraudulent.")

time = st.number_input("Transaction Time", value=1000.0, step=1.0)
amount = st.number_input("Transaction Amount", value=50.0, step=0.1)

features = {}
for i in range(1, 29):
    features[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.1)

if st.button("🔍 Predict Fraud"):
    data = np.array([[
        time,
        *[features[f"V{i}"] for i in range(1, 29)],
        amount
    ]])

    data[:, [0]] = scaler.transform(data[:, [0]])
    data[:, [-1]] = scaler.transform(data[:, [-1]])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Legitimate Transaction (Fraud Probability: {probability:.2%})")


