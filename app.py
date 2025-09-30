import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="🔧 Predictive Maintenance",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🔧 Predictive Maintenance Dashboard")
st.write("Upload IoT sensor data to predict failures using the LSTM model.")

# ---------------- Load LSTM Model ----------------
lstm_model = load_model("lstm_failure_model.keras")

# ---------------- Load Feature List ----------------
with open("features.pkl", "rb") as f:
    features = pickle.load(f)

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload IoT Dataset CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # ---------------- Display Data (Compact) ----------------
    st.subheader("Uploaded Data (First 10 rows)")
    st.dataframe(data.head(10), height=200)  # smaller table height

    # ---------------- Align Features ----------------
    for col in features:
        if col not in data.columns:
            data[col] = 0
    X = data[features].fillna(0)
    X_scaled = X.values

    # ---------------- LSTM Prediction ----------------
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    lstm_preds = lstm_model.predict(X_lstm)
    lstm_preds = (lstm_preds > 0.5).astype(int)
    data["LSTM_Prediction"] = lstm_preds

    st.subheader("Predictions (First 10 rows)")
    st.dataframe(data[["LSTM_Prediction"]].head(10), height=150)  # compact table

    # ---------------- Prediction Distribution (Compact) ----------------
    st.subheader("📊 Prediction Distribution (LSTM)")
    fig, ax = plt.subplots(figsize=(4,3))  # smaller figure
    sns.countplot(x='LSTM_Prediction', data=data, palette='coolwarm', ax=ax)
    ax.set_title("Predicted Failures vs Non-Failures", fontsize=12)
    ax.set_xlabel("Prediction", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + 0.25, p.get_height() + 0.3), fontsize=9)
    st.pyplot(fig)

    # ---------------- Pie Chart (Compact) ----------------
    st.subheader("🥧 Failure Proportion")
    counts = data["LSTM_Prediction"].value_counts().sort_index()
    labels = ['No Failure' if i == 0 else 'Failure' for i in counts.index]
    colors = ['#4CAF50' if i == 0 else '#F44336' for i in counts.index]

    fig2, ax2 = plt.subplots(figsize=(3,3))  # smaller pie chart
    ax2.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, shadow=True)
    ax2.set_title("Failure vs Non-Failure Ratio", fontsize=12)
    st.pyplot(fig2)

    # ---------------- Correlation Heatmap (Compact) ----------------
    st.subheader("🌡 Feature Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(6,5))  # smaller heatmap
    corr = data[features].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax3)
    ax3.set_title("Feature Correlation", fontsize=12)
    st.pyplot(fig3)

    # ---------------- Alerts ----------------
    failure_count = int(sum(lstm_preds))
    if failure_count > 0:
        st.error(f"{failure_count} machines at risk of failure (LSTM)!")
    else:
        st.success("All machines running normally (LSTM).")

    # ---------------- Download CSV ----------------
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Dataset with Predictions",
        data=csv,
        file_name="iot_lstm_predictions.csv",
        mime="text/csv"
    )
