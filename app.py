
<style>
    /* MAIN PAGE */
    .stApp {
        background-color: #FFFFFF;
        color: #292929;
    }

    [data-testid="stHeader"] {
        background-color: #FFFFFF;
    }

    .block-container {
        max-width: 850px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* HEADINGS */
    h1, h2, h3 {
        color: #292929 !important;
    }

    h1 {
        font-size: 2rem !important;
        font-weight: 650 !important;
    }

    /* SUBTITLE AND HELPER TEXT */
    .subtitle, .helper-text {
        color: #666666 !important;
    }

    /* MODEL INFORMATION */
    .model-note {
        background-color: #F8F9FA;
        border: 1px solid #E5E5E5;
        border-left: 4px solid #8B3A45;
        padding: 14px;
        color: #292929;
        border-radius: 6px;
    }

    /* SECTION HEADINGS */
    .section-heading {
        color: #292929;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }

    /* INPUT FORM */
    div[data-testid="stForm"] {
        background-color: #F8F9FA;
        border: 1px solid #E5E5E5;
        border-radius: 8px;
        padding: 1.25rem;
    }

    /* INPUT LABELS */
    div[data-testid="stNumberInput"] label {
        color: #292929 !important;
        font-weight: 500;
    }

    /* INPUT FIELDS */
    div[data-testid="stNumberInput"] input {
        background-color: #FFFFFF !important;
        color: #292929 !important;
        border: 1px solid #DDDDDD !important;
        border-radius: 5px;
    }

    /* INPUT CONTROLS */
    div[data-testid="stNumberInput"] button {
        color: #8B3A45 !important;
    }

    /* PREDICT BUTTON */
    div[data-testid="stFormSubmitButton"] button {
        background-color: #8B3A45;
        color: #FFFFFF;
        border: 1px solid #8B3A45;
        border-radius: 5px;
        font-weight: 600;
        min-height: 2.7rem;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #71303A;
        border-color: #71303A;
        color: #FFFFFF;
    }

    /* PREDICTION RESULT */
    div[data-testid="stAlert"] {
        background-color: #F8F9FA !important;
        border: 1px solid #E5E5E5 !important;
        border-left: 4px solid #8B3A45 !important;
        border-radius: 6px;
    }

    div[data-testid="stAlert"] p {
        color: #292929 !important;
    }

    /* PROBABILITY METRICS */
    div[data-testid="stMetric"] {
        background-color: #F8F9FA;
        border: 1px solid #E5E5E5;
        border-radius: 6px;
        padding: 12px;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricValue"] {
        color: #292929 !important;
    }

    /* EXPANDABLE SECTIONS */
    div[data-testid="stExpander"] {
        background-color: #F8F9FA;
        border: 1px solid #E5E5E5;
        border-radius: 6px;
    }

    div[data-testid="stExpander"] summary {
        color: #292929 !important;
    }

    /* TABLES */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E5E5E5;
        border-radius: 5px;
    }

    /* DIVIDERS */
    hr {
        border-color: #E5E5E5 !important;
    }

    /* FOOTER */
    .footer-note {
        color: #777777 !important;
        text-align: center;
        font-size: 0.78rem;
        margin-top: 2rem;
    }
</style>
