import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# ---------------- Page Config ----------------
st.set_page_config(page_title="🔧 Predictive Maintenance", layout="centered")
st.title(" Predictive Maintenance Dashboard")
st.write("Upload IoT sensor data to predict failures, RUL, and visualize insights.")

# ---------------- Load Models & Scalers ----------------
xgb_model = joblib.load("xgb_failure_model.pkl")
rf_model = joblib.load("rf_failure_model.pkl")
lstm_model = load_model("lstm_failure_model.keras")
rul_model = joblib.load("rul_model.pkl")

scaler = joblib.load("scaler.pkl")
scaler_rul = joblib.load("scaler_rul.pkl")

features = joblib.load("features.pkl")
features_rul = joblib.load("features_rul.pkl")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload IoT Dataset CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader(" Uploaded Data (First 10 rows)")
    st.dataframe(data.head(10))

    # ---------------- Rolling Mean Features ----------------
    original_features = ['Temperature','Vibration','Pressure','Voltage','Current',
                         'FFT_Feature1','FFT_Feature2','Normalized_Temp','Normalized_Vibration',
                         'Normalized_Pressure','Normalized_Voltage','Normalized_Current','Anomaly_Score',
                         'Fault_Status']
    for col in original_features:
        rolling_col = f"{col}_rolling_mean"
        if col in data.columns and rolling_col not in data.columns:
            data[rolling_col] = data[col].rolling(window=5, min_periods=1).mean()

    # ---------------- Align Features ----------------
    for col in features:
        if col not in data.columns:
            data[col] = 0
    X = data[features].fillna(0)
    X_scaled = scaler.transform(X)

    for col in features_rul:
        if col not in data.columns:
            data[col] = 0
    X_rul = data[features_rul].fillna(0)
    X_rul_scaled = scaler_rul.transform(X_rul)

    # ---------------- Predictions ----------------
    xgb_preds = xgb_model.predict(X)
    rf_preds = rf_model.predict(X)
    
    # LSTM requires 3D input
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    lstm_preds = lstm_model.predict(X_lstm)
    lstm_preds = (lstm_preds > 0.5).astype(int)
    
    rul_preds = rul_model.predict(X_rul_scaled)

    # ---------------- Add Predictions ----------------
    data["XGB_Prediction"] = xgb_preds
    data["RF_Prediction"] = rf_preds
    data["LSTM_Prediction"] = lstm_preds
    data["RUL_Prediction"] = rul_preds

    st.subheader(" Predictions (First 10 rows)")
    st.dataframe(data[["XGB_Prediction","RF_Prediction","LSTM_Prediction","RUL_Prediction"]].head(10))

    # ---------------- Combined Prediction Distribution ----------------
    st.subheader(" Prediction Distributions (All Models)")
    pred_df = data[['XGB_Prediction','RF_Prediction','LSTM_Prediction']].melt(
        var_name='Model', value_name='Prediction'
    )
    fig, ax = plt.subplots(figsize=(6,3))
    sns.countplot(x='Prediction', hue='Model', data=pred_df, palette='Set2', ax=ax)
    ax.set_title("Predicted Failures vs Non-Failures (All Models)")
    st.pyplot(fig)

    # ---------------- RUL Distribution ----------------
    st.subheader(" Remaining Useful Life (RUL) Distribution")
    fig2, ax2 = plt.subplots(figsize=(6,3))
    sns.histplot(data["RUL_Prediction"], bins=20, kde=True, color="green", ax=ax2)
    ax2.set_title("Predicted Remaining Useful Life")
    st.pyplot(fig2)

    # ---------------- Alerts ----------------
    failure_count = int(sum(xgb_preds))
    if failure_count > 0:
        st.error(f" {failure_count} machines at risk of failure (XGBoost)!")
    else:
        st.success("All machines running normally (XGBoost).")

    # ---------------- Download CSV ----------------
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Dataset with All Predictions",
        data=csv,
        file_name="iot_full_predictions.csv",
        mime="text/csv"
    )
