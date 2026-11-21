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
