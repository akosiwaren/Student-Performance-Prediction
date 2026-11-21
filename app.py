
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_success_gb_model.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------
# LIGHT MODE DESIGN
# ------------------------------------------

st.markdown(
    """


<style>
    .stApp {
        background-color: #15345B;
        color: #F1FAEF;
    }

    [data-testid="stHeader"] {
        background-color: #15345B;
    }

    .block-container {
        max-width: 850px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, p {
        color: #F1FAEF !important;
    }

    h1 {
        font-size: 2rem !important;
        font-weight: 650 !important;
    }

    .subtitle, .helper-text {
        color: #ACD9DD !important;
    }

    .model-note {
        background-color: #4779A1;
        border-left: 4px solid #ACD9DD;
        padding: 14px;
        color: #FFFFFF;
        border-radius: 5px;
    }

    div[data-testid="stForm"] {
        background-color: #1D4268;
        border: 1px solid #4779A1;
        border-radius: 7px;
        padding: 1.25rem;
    }

    div[data-testid="stNumberInput"] label {
        color: #F1FAEF !important;
        font-weight: 500;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #F5F4F4 !important;
        color: #15345B !important;
        border: 1px solid #ACD9DD !important;
    }

    div[data-testid="stFormSubmitButton"] button {
        background-color: #ACD9DD;
        color: #15345B;
        border: 1px solid #ACD9DD;
        border-radius: 5px;
        font-weight: 600;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #F1FAEF;
        border-color: #F1FAEF;
        color: #15345B;
    }

    div[data-testid="stExpander"] {
        background-color: #1D4268;
        border: 1px solid #4779A1;
        border-radius: 6px;
    }

    div[data-testid="stExpander"] summary {
        color: #F1FAEF !important;
    }

    .footer-note {
        color: #ACD9DD !important;
        text-align: center;
        font-size: 0.78rem;
        margin-top: 2rem;
    }
</style>


    """,
    unsafe_allow_html=True
)

# ------------------------------------------
# LOAD MODEL AND METADATA
# ------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

model = load_model()

# ------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------

def engineer_features(raw):

    result = raw.copy()

    active = (
        result["Submission.Actions"]
        + result["Quiz.Actions"]
        + result["Forum.Actions"]
        + result["Create.Actions"]
    )

    passive = (
        result["Read.Actions"]
        + result["Update.Actions"]
        + result["Delete.Actions"]
    )

    total = active + passive

    days = result["Days.Logged.In"].replace(0, 1)

    result["Total_Actions"] = total
    result["Active_Actions"] = active
    result["Passive_Actions"] = passive

    result["Actions_Per_Day"] = total / days
    result["Active_Per_Day"] = active / days
    result["Active_Ratio"] = active / (total + 1)

    result["Quiz_Submission_Sum"] = (
        result["Quiz.Actions"]
        + result["Submission.Actions"]
    )

    result["Log_Total_Actions"] = np.log1p(total)
    result["Log_Days_Logged_In"] = np.log1p(
        result["Days.Logged.In"]
    )

    return result[metadata["engineered_features"]]


# ------------------------------------------
# APPLICATION HEADER
# ------------------------------------------

st.title("Student Performance Predictor")

st.markdown(
    """
    <p class="subtitle">
    Predict student midterm performance using
    Learning Management System (LMS) activity data.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="model-note">
    <strong>Machine Learning Model:</strong>
    This application uses the Gradient Boosting Classifier
    developed and trained in Machine Problem 2 (MP2).
    The original model and its preprocessing workflow
    are retained for this application.
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------
# STUDENT INPUT FORM
# ------------------------------------------

st.markdown(
    '<p class="section-heading">Student Activity Information</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="helper-text">
    Enter the student's LMS activity counts.
    The application automatically calculates the
    required behavioral predictors.
    </p>
    """,
    unsafe_allow_html=True
)

labels = {
    "Submission.Actions": "Submission Actions",
    "Quiz.Actions": "Quiz Actions",
    "Forum.Actions": "Forum Actions",
    "Create.Actions": "Create Actions",
    "Read.Actions": "Read Actions",
    "Update.Actions": "Update Actions",
    "Delete.Actions": "Delete Actions",
    "Days.Logged.In": "Days Logged In"
}

defaults = {
    "Submission.Actions": 10,
    "Quiz.Actions": 5,
    "Forum.Actions": 2,
    "Create.Actions": 10,
    "Read.Actions": 100,
    "Update.Actions": 5,
    "Delete.Actions": 0,
    "Days.Logged.In": 10
}

with st.form("prediction_form"):

    left, right = st.columns(2)

    values = {}

    for i, feature in enumerate(metadata["raw_features"]):

        column = left if i % 2 == 0 else right

        with column:

            values[feature] = st.number_input(
                labels[feature],
                min_value=0,
                value=defaults[feature],
                step=1
            )

    submitted = st.form_submit_button(
        "Predict Performance",
        type="primary",
        use_container_width=True
    )


# ------------------------------------------
# PREDICTION
# ------------------------------------------

if submitted:

    raw_input = pd.DataFrame(
        [values],
        columns=metadata["raw_features"]
    )

    model_input = engineer_features(raw_input)

    prediction = model.predict(model_input)[0]

    probabilities = model.predict_proba(model_input)[0]

    class_probabilities = dict(
        zip(model.classes_, probabilities)
    )

    high_prob = float(
        class_probabilities.get("High", 0)
    )

    low_prob = float(
        class_probabilities.get("Low", 0)
    )

    st.divider()

    st.markdown(
        '<p class="section-heading">Prediction Result</p>',
        unsafe_allow_html=True
    )

    if prediction == "High":

        st.success("Predicted Category: High Performance")

    else:

        st.warning("Predicted Category: Low Performance")

    col1, col2 = st.columns(2)

    col1.metric(
        "High Performance Probability",
        f"{high_prob * 100:.1f}%"
    )

    col2.metric(
        "Low Performance Probability",
        f"{low_prob * 100:.1f}%"
    )

    with st.expander("View Calculated Predictors"):

        st.dataframe(
            model_input.T.rename(
                columns={0: "Value"}
            ),
            use_container_width=True
        )

    st.caption(
        "This prediction is an early-warning estimate "
        "based on LMS activity. It should support academic "
        "intervention and not determine a student's grade."
    )


# ------------------------------------------
# MODEL INFORMATION
# ------------------------------------------

with st.expander("Model Details and Evaluation"):

    st.write("**Algorithm:** Gradient Boosting Classifier")

    st.write("**Model Source:** Machine Problem 2 (MP2)")

    st.write("**Raw Input Variables:** 8")

    st.write("**Engineered Variables:** 7")

    st.write("**Total Predictors:** 15")

    st.write("**Train-Test Split:** 80% / 20%")

    st.write(
        f"**Training Records:** {metadata['training_rows']:,}"
    )

    st.write(
        f"**Testing Records:** {metadata['testing_rows']:,}"
    )

    st.write(
        f"**Holdout Accuracy:** "
        f"{metadata['accuracy'] * 100:.2f}%"
    )

    st.write(
        f"**ROC-AUC:** {metadata['roc_auc']:.3f}"
    )

    st.write("**Classification Report**")

    report_df = pd.DataFrame(
        metadata["classification_report"]
    ).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.write("**Confusion Matrix**")

    confusion_df = pd.DataFrame(
        metadata["confusion_matrix"],
        index=["Actual High", "Actual Low"],
        columns=["Predicted High", "Predicted Low"]
    )

    st.dataframe(
        confusion_df,
        use_container_width=True
    )


# ------------------------------------------
# FOOTER
# ------------------------------------------

st.markdown(
    """
    <p class="footer-note">
    Machine Problem 3 | Student Performance Prediction System
    </p>
    """,
    unsafe_allow_html=True
)
