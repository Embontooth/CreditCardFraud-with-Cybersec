import joblib
import streamlit as st
import pandas as pd

rf = joblib.load("models/rf_model.pkl")
iso = joblib.load("models/iso_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Cyber-Financial Fraud Detection Dashboard")
st.write("Detect suspicious credit card transactions with AI + Cybersecurity signals")

with st.form(key='txn_form'):
    st.header("Enter Transaction Details")
    v_features = {}
    for i in range(1, 29):
        v_features[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.01)

    time = st.number_input("Time", value=0)
    amount = st.number_input("Amount", value=0.0)

    ip_risk_score = st.slider("IP Risk Score", 0.0, 1.0, 0.1)
    failed_login_count = st.number_input("Failed Login Count", 0, 10, 0)
    device_change = st.selectbox("Device Changed?", [0, 1])
    vpn_usage = st.selectbox("VPN Usage?", [0, 1])

    submit_button = st.form_submit_button(label='Submit Transaction')

if submit_button:
    txn = pd.DataFrame({
        **v_features,
        "Time": [time],
        "Amount": [amount],
        "ip_risk_score": [ip_risk_score],
        "failed_login_count": [failed_login_count],
        "device_change": [device_change],
        "vpn_usage": [vpn_usage]
    })

    txn_scaled = scaler.transform(txn)

    if iso.predict(txn_scaled) == -1:
        st.warning("⚠️ Transaction flagged as SUSPICIOUS")
        fraud_pred = rf.predict(txn_scaled)
        if fraud_pred[0] == 1:
            st.error("❌ Fraudulent Transaction Detected!")
        else:
            st.success("✔ Suspicious but likely NOT fraud")
    else:
        st.success("✔ Transaction looks normal")
