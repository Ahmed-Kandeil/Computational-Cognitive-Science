import streamlit as st
import pandas as pd

from models.mix_naive_bayes_classifier import mix_naive_bayes_classifier


st.set_page_config("Mix Naive Bayes Classifier", page_icon="🧮")
st.title("Mix Naive Bayes Classifier")


def read_data(file_path) -> pd.DataFrame:
    return (
        pd.read_csv(file_path)
        if file_path.name.endswith(".csv")
        else pd.read_excel(file_path)
    )


data = st.file_uploader("Upload data file", type=["csv", "xlsx"])

if data:
    df = read_data(data)
    features = df.columns.tolist()

    st.dataframe(df)

    feature_container, target_container = st.columns([0.7, 0.3])

    with feature_container:
        feature_cols = st.multiselect(
            "Feature Columns",
            options=features,
            default=features[:-1],
            max_selections=len(features) - 1,
        )

    with target_container:
        target_col = st.selectbox(
            "Target Column", options=features, index=len(features) - 1
        )

    gaussian_container, exponential_container = st.columns(2)

    with gaussian_container:
        gaussian_cols = st.multiselect(
            "Gaussian Columns",
            options=feature_cols,
            max_selections=len(feature_cols),
        )

    remaining_feature_cols = [
        feature_col for feature_col in feature_cols if feature_col not in gaussian_cols
    ]

    with exponential_container:
        exponential_cols = st.multiselect(
            "Exponential Columns",
            options=remaining_feature_cols,
            max_selections=len(feature_cols),
        )

    values_container, std_container = st.columns([0.7, 0.3])

    default_len = len(gaussian_cols) + len(exponential_cols)
    default_values = ", ".join(["0" for _ in range(default_len)])

    with values_container:
        values_input = st.text_input("Values", value=default_values)

    with std_container:
        std = st.text_input("Standard Deviation")

    if values_input:
        values = [int(value) for value in values_input.split(",")]

    if st.button("Calc"):
        result_df, max_prob_class = mix_naive_bayes_classifier(
            df, gaussian_cols, exponential_cols, target_col, values, float(std)
        )

        st.dataframe(result_df)
        st.success(f"The {values} belong to class {max_prob_class}")
        st.balloons()
