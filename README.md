# 💳 Fraud Detection in Transactions (ML-based)

## 📌 Overview
- Built a fraud detection system using **Scikit-learn + XGBoost** on transaction datasets.
- Improved anomaly detection accuracy by **20%** through feature engineering and ensemble learning.
- Deployed a prototype using **Streamlit**, suitable for real-time fraud monitoring demo.

## ⚙️ How it Works
1. Preprocess transaction dataset (`Time`, `Amount`, PCA components V1–V28).
2. Handle class imbalance with **SMOTE**.
3. Train baseline Logistic Regression → then improve with **XGBoost**.
4. Save trained model + scaler (`saved_models/`).
5. Launch Streamlit app → predict whether a transaction is fraudulent.

## 🚀 Run Locally
```bash
git clone https://github.com/RITWIK4959/Fraud_Detection.git
cd Fraud_Detection
pip install -r requirements.txt
streamlit run app.py
