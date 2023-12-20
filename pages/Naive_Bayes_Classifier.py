import streamlit as st
import pandas as pd

from models.naive_bayes_classifier import naive_bayes_classifier


st.set_page_config("Naive Bayes Classifier", page_icon="🧮")


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

    default_values = ", ".join(["0" for _ in range(len(feature_cols))])
    values_input = st.text_input("Values", value=default_values)

    if values_input:
        values = [int(value) for value in values_input.split(",")]

    if st.button("Calc"):
        result_df, max_prob_class = naive_bayes_classifier(
            df, feature_cols, target_col, values
        )

        st.dataframe(result_df)
        st.success(f"The {values} belong to class {max_prob_class}")
        st.balloons()
