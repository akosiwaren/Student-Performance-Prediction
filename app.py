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
