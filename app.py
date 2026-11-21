
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
    layout="centered"
)

# ------------------------------------------
# CUSTOM DESIGN
# ------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            max-width: 850px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        h1 {
            font-size: 2rem !important;
            font-weight: 650 !important;
            letter-spacing: -0.5px;
        }

        .subtitle {
            color: #667085;
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }

        .section-heading {
            font-size: 1.15rem;
            font-weight: 600;
            margin-top: 1.2rem;
            margin-bottom: 0.3rem;
        }

        .helper-text {
            color: #667085;
            font-size: 0.88rem;
            margin-bottom: 1rem;
        }

        div[data-testid="stForm"] {
            border: 1px solid #D0D5DD;
            border-radius: 8px;
            padding: 1.25rem;
        }

        div[data-testid="stNumberInput"] label {
            font-weight: 500;
        }

        div[data-testid="stFormSubmitButton"] button {
            border-radius: 6px;
            font-weight: 600;
            min-height: 2.7rem;
        }

        div[data-testid="stExpander"] {
            border: 1px solid #D0D5DD;
            border-radius: 8px;
        }

        .footer-note {
            color: #98A2B3;
            font-size: 0.78rem;
            text-align: center;
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

    st.write("**Raw Input Variables:** 8")

    st.write("**Engineered Variables:** 7 as described in MP2")

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
    Machine Problem 2 | Student Performance Prediction System
    </p>
    """,
    unsafe_allow_html=True
)
