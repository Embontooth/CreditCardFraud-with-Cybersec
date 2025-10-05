import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/creditcard.csv")


df["ip_risk_score"] = np.random.rand(len(df))
df["failed_login_count"] = np.random.randint(0, 5, len(df))
df["device_change"] = np.random.randint(0, 2, len(df))
df["vpn_usage"] = np.random.randint(0, 2, len(df))

features = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "ip_risk_score", "failed_login_count", "device_change", "vpn_usage"]
X = df[features]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_scaled, y_train)

# y_pred = model.predict(X_test_scaled)

# print(classification_report(y_test, y_pred))


iso = IsolationForest(contamination=0.001, random_state=42)
iso.fit(X_train_scaled)

outliers = iso.predict(X_test_scaled)

sus_data = X_test_scaled[outliers==-1]
sus_lab = y_test[outliers==-1].values

rf = RandomForestClassifier(n_estimators=200, random_state=42,verbose=1, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

fraud_preds = rf.predict(sus_data)

# print(classification_report(sus_lab, fraud_preds))

joblib.dump(rf, "models/rf_model.pkl")
joblib.dump(iso, "models/iso_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Models saved successfully!")