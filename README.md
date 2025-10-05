# Cyber-Financial Fraud Detection System

A machine learning-powered fraud detection system that combines traditional credit card transaction analysis with cybersecurity indicators to identify suspicious and fraudulent activities.

## Overview

This project implements a two-stage fraud detection pipeline:
1. **Anomaly Detection**: Uses Isolation Forest to flag suspicious transactions
2. **Fraud Classification**: Applies Random Forest to classify flagged transactions as fraudulent or legitimate

The system enhances traditional financial features with cybersecurity signals like IP risk scores, failed login attempts, device changes, and VPN usage patterns.

## Features

- **Dual-Stage Detection**: Anomaly detection followed by classification for improved accuracy
- **Cybersecurity Integration**: Incorporates behavioral and network security indicators
- **Interactive Dashboard**: Streamlit web interface for real-time transaction analysis
- **Model Persistence**: Saves trained models for reuse and deployment
- **Comprehensive Analysis**: Handles 28 PCA-transformed features plus custom security metrics

## Technology Stack

- **Machine Learning**: scikit-learn (Isolation Forest, Random Forest)
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib
- **Visualization**: Streamlit components
  
##  Dataset

The system expects a credit card dataset with the following structure:
- **V1-V28**: PCA-transformed features (anonymized)
- **Time**: Transaction timestamp
- **Amount**: Transaction amount
- **Class**: Target variable (0=legitimate, 1=fraud)

Additional cybersecurity features are generated:
- **ip_risk_score**: IP address risk assessment (0-1)
- **failed_login_count**: Number of recent failed login attempts
- **device_change**: Whether device was changed (0/1)
- **vpn_usage**: VPN usage indicator (0/1)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AI
   ```

2. **Install dependencies**:
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib
   ```

3. **Prepare the dataset**:
   - Place your `creditcard.csv` file in the project root
   - Ensure it contains the required columns (V1-V28, Time, Amount, Class)

4. **Create models directory**:
   ```bash
   mkdir models
   ```

## Usage

### Training Models

Run the main script to train and save models:
```bash
python main.py
```

This will:
- Load and preprocess the credit card dataset
- Generate synthetic cybersecurity features
- Train Isolation Forest and Random Forest models
- Save models to the `models/` directory

### Running the Dashboard

Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```

The dashboard provides:
- **Input Panel**: Enter transaction details and cybersecurity indicators
- **Real-time Analysis**: Instant fraud risk assessment
- **Visual Feedback**: Color-coded alerts and status messages

## Model Architecture

### Stage 1: Anomaly Detection
- **Algorithm**: Isolation Forest
- **Purpose**: Identify transactions that deviate from normal patterns
- **Contamination Rate**: 0.1% (adjustable)
- **Features**: All 32 features (V1-V28 + Time, Amount + 4 cyber features)

### Stage 2: Fraud Classification
- **Algorithm**: Random Forest
- **Purpose**: Classify anomalous transactions as fraud/legitimate
- **Estimators**: 200 trees
- **Input**: Only transactions flagged as suspicious by Stage 1

### Feature Engineering
- **Standardization**: StandardScaler normalization
- **Synthetic Features**: Cybersecurity indicators added to enhance detection
- **Stratified Split**: Maintains class balance in train/test sets

## 📈 Performance Considerations

- **Training Time**: Models retrain on every app reload (consider caching for production)
- **Memory Usage**: Random Forest with 200 estimators requires significant RAM
- **Scalability**: Current implementation suitable for datasets up to ~100K transactions

## Security Features

The system incorporates multiple cybersecurity indicators:

| Feature | Description | Range |
|---------|-------------|-------|
| IP Risk Score | Reputation-based IP assessment | 0.0 - 1.0 |
| Failed Login Count | Recent authentication failures | 0 - 10+ |
| Device Change | New device detection | 0/1 |
| VPN Usage | VPN/proxy detection | 0/1 |

## Customization

### Adjusting Detection Sensitivity
Modify the contamination parameter in `main.py`:
```python
iso = IsolationForest(contamination=0.001, random_state=42)  # 0.1% contamination
```

### Adding New Features
Extend the features list:
```python
features = [f"V{i}" for i in range(1, 29)] + [
    "Time", "Amount", "ip_risk_score", "failed_login_count", 
    "device_change", "vpn_usage", "your_new_feature"
]
```

### Model Hyperparameters
Tune Random Forest settings:
```python
rf = RandomForestClassifier(
    n_estimators=200,      # Number of trees
    max_depth=None,        # Tree depth
    min_samples_split=2,   # Split threshold
    random_state=42
)
```
## Disclaimer

This system is for educational and research purposes. For production fraud detection, implement additional security measures, data validation, and regulatory compliance checks.

## Future Enhancements

- [ ] Real-time model updates
- [ ] Advanced feature engineering
- [ ] Model performance monitoring
- [ ] API endpoint for integration
- [ ] Docker containerization
- [ ] Database integration for transaction history
- [ ] Explainable AI features for decision transparency
