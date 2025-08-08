# monitoring/dashboard.py

import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score

LOG_FILE = "logs/prediction_logs.json"
DATASET_FILE = "IMDB_Dataset.csv"

# Load logs
def load_logs():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

# Load dataset
def load_dataset():
    return pd.read_csv(DATASET_FILE)

# Sentence length helper
def sentence_lengths(texts):
    return [len(text.split()) for text in texts]

# Streamlit app
st.title("📊 Streamlit Monitoring Dashboard")

# 0. Read everything
logs = load_logs()
if not logs:
    st.warning("Waiting for predictions to be logged…")
    st.stop()

df_logs = pd.DataFrame(logs)
df_dataset = load_dataset()

# 1. Data Drift
st.subheader("1. Data Drift (Sentence Lengths)")
logged_lengths  = sentence_lengths(df_logs["request_text"])
dataset_lengths = sentence_lengths(df_dataset["review"])

plt.figure()
plt.hist(dataset_lengths, bins=30, alpha=0.5, label="Original Dataset")
plt.hist(logged_lengths,  bins=30, alpha=0.5, label="Logged Requests")
plt.legend()
plt.xlabel("Sentence Length")
plt.ylabel("Frequency")
st.pyplot(plt)

# 2. Target Drift
st.subheader("2. Target Drift (Prediction Frequency)")
pred_counts = df_logs["predicted_sentiment"].value_counts().sort_index()
st.bar_chart(pred_counts)

# 3. Accuracy & Precision (only if we have true labels)
if "true_sentiment" in df_logs.columns and df_logs["true_sentiment"].notna().any():
    # ─── MAPPING STRINGS TO INTEGERS ────────────────────────────────────────────────
    mapping = {"negative": 0, "positive": 1}
    # map both columns
    df_logs["true_sentiment_num"]      = df_logs["true_sentiment"].map(mapping)
    df_logs["predicted_sentiment_num"] = df_logs["predicted_sentiment"].map(mapping)
    # ────────────────────────────────────────────────────────────────────────────────

    y_true = df_logs["true_sentiment_num"]
    y_pred = df_logs["predicted_sentiment_num"]

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    st.subheader("3. Model Accuracy & Feedback")
    st.write(f"**Accuracy:**  {accuracy:.2%}")
    st.write(f"**Precision:** {precision:.2%}")

    # 4. Alert
    if accuracy < 0.80:
        st.error("⚠️ Model accuracy has dropped below 80%!")

else:
    st.info("Waiting for true sentiment labels to compute metrics.")
