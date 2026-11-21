import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_success_gb_model.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"

st.set_page_config(page_title="MP2 Student Performance Predictor", page_icon="🎓", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

model = load_model()

def engineer_features(raw):
    result = raw.copy()
    active = (
        result["Submission.Actions"] + result["Quiz.Actions"] +
        result["Forum.Actions"] + result["Create.Actions"]
    )
    passive = result["Read.Actions"] + result["Update.Actions"] + result["Delete.Actions"]
    total = active + passive
    result["Total_Actions"] = total
    result["Active_Actions"] = active
    result["Passive_Actions"] = passive
    result["Actions_Per_Day"] = total / result["Days.Logged.In"].replace(0, 1)
    result["Active_Per_Day"] = active / result["Days.Logged.In"].replace(0, 1)
    result["Active_Ratio"] = active / (total + 1)
    result["Quiz_Submission_Sum"] = result["Quiz.Actions"] + result["Submission.Actions"]
    result["Log_Total_Actions"] = np.log1p(total)
    result["Log_Days_Logged_In"] = np.log1p(result["Days.Logged.In"])
    return result[metadata["engineered_features"]]

st.title("🎓 Student Performance Predictor")
st.write("Predict a student's midterm performance category using Learning Management System (LMS) activity.")

st.info(
    "This application follows the preprocessing and Gradient Boosting workflow from Machine Problem 2. "
    "The model predicts **High** or **Low** performance; it does not directly predict final pass/fail status."
)

st.subheader("Enter Student LMS Activity")
st.caption("Enter the student's raw activity counts. The application automatically calculates the seven engineered features.")

with st.form("prediction_form"):
    defaults = {
        "Submission.Actions": 10,
        "Quiz.Actions": 5,
        "Forum.Actions": 2,
        "Create.Actions": 10,
        "Read.Actions": 100,
        "Update.Actions": 5,
        "Delete.Actions": 0,
        "Days.Logged.In": 10,
    }
    labels = {
        "Submission.Actions": "Submission Actions",
        "Quiz.Actions": "Quiz Actions",
        "Forum.Actions": "Forum Actions",
        "Create.Actions": "Create Actions",
        "Read.Actions": "Read Actions",
        "Update.Actions": "Update Actions",
        "Delete.Actions": "Delete Actions",
        "Days.Logged.In": "Days Logged In",
    }
    values = {}
    left, right = st.columns(2)
    for i, feature in enumerate(metadata["raw_features"]):
        with (left if i % 2 == 0 else right):
            values[feature] = st.number_input(
                labels[feature], min_value=0, value=defaults[feature], step=1
            )
    submitted = st.form_submit_button("Predict Performance", type="primary", use_container_width=True)

if submitted:
    raw_input = pd.DataFrame([values], columns=metadata["raw_features"])
    engineered_input = engineer_features(raw_input)
    prediction = model.predict(engineered_input)[0]
    probabilities = model.predict_proba(engineered_input)[0]
    class_probabilities = dict(zip(model.classes_, probabilities))

    st.divider()
    st.subheader("Prediction Result")
    if prediction == "High":
        st.success("Predicted Category: HIGH PERFORMANCE")
    else:
        st.warning("Predicted Category: LOW PERFORMANCE — Consider early academic support.")

    high_prob = float(class_probabilities.get("High", 0))
    low_prob = float(class_probabilities.get("Low", 0))
    c1, c2 = st.columns(2)
    c1.metric("High probability", f"{high_prob * 100:.1f}%")
    c2.metric("Low probability", f"{low_prob * 100:.1f}%")
    st.progress(high_prob, text=f"High: {high_prob * 100:.1f}%")
    st.progress(low_prob, text=f"Low: {low_prob * 100:.1f}%")

    with st.expander("View calculated predictors"):
        st.dataframe(engineered_input.T.rename(columns={0: "Value"}), use_container_width=True)

    st.caption("This prediction is an early-warning estimate based on LMS activity patterns, not a final grade or a basis for penalizing a student.")

with st.expander("Model details and evaluation"):
    st.write("**Algorithm:** Stochastic Gradient Boosting Classifier")
    st.write("**Raw input variables:** 8")
    st.write("**Engineered predictors:** 7")
    st.write("**Total model predictors:** 15")
    st.write("**Training/testing split:** 80% / 20% (stratified)")
    st.write(f"**Training records:** {metadata['training_rows']:,}")
    st.write(f"**Testing records:** {metadata['testing_rows']:,}")
    st.write(f"**Holdout accuracy:** {metadata['accuracy'] * 100:.2f}%")
    st.write(f"**ROC-AUC:** {metadata['roc_auc']:.3f}")
    st.write("**Classification report:**")
    st.dataframe(pd.DataFrame(metadata["classification_report"]).transpose().round(3), use_container_width=True)
    st.write("**Confusion matrix** (rows = actual High/Low; columns = predicted High/Low):")
    st.dataframe(pd.DataFrame(metadata["confusion_matrix"], index=["Actual High", "Actual Low"], columns=["Predicted High", "Predicted Low"]), use_container_width=True)

st.caption("Machine Problem 2 | Student Performance Early-Warning Application")
