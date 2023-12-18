import streamlit as st
import pandas as pd

from models.posterior_probabilities import process_dataframe, posterior_probabilities


st.set_page_config("Evidences", page_icon="🧮")


def read_data(file_path) -> pd.DataFrame:
    return (
        pd.read_csv(file_path)
        if file_path.name.endswith(".csv")
        else pd.read_excel(file_path)
    )


data = st.file_uploader("Upload data file", type=["csv", "xlsx"])

if data:
    df = read_data(data)
    df = process_dataframe(df)

    evidences = df.index.to_list()[1:]
    hypothesis = df.columns.tolist()

    st.dataframe(df)

    evidences_container, hypothesis_container = st.columns([0.55, 0.45])

    with evidences_container:
        evidences = st.multiselect(
            "Evidences",
            options=evidences,
            default=evidences,
        )

    with hypothesis_container:
        hypothesis = st.multiselect(
            "Hypothesis",
            options=hypothesis,
            default=hypothesis,
        )

    if st.button("Calc"):
        result_df, max_posterior_probability = posterior_probabilities(
            df, evidences, hypothesis
        )

        st.dataframe(result_df)
        st.success(
            f"The max posterior probability belong to {max_posterior_probability}"
        )
        st.balloons()
