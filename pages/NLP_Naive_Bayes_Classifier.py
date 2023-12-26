import streamlit as st
import pandas as pd

from models.nlp_naive_bayes_classifier import nlp


st.set_page_config("NLP NLP", page_icon="🧮")
st.title("NLP Naive Bayes Classifier")


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

    text_container, tag_container = st.columns(2)

    with text_container:
        text_col = st.selectbox("Text Column", options=features, index=0)

    with tag_container:
        tag_col = st.selectbox("Tag Column", options=features, index=len(features) - 1)

    sentence_input = st.text_input("Sentence")
    tokenization = st.toggle("Tokenization")

    if st.button("Calc"):
        result_df, max_prob_class = nlp(
            df,
            text_col,
            tag_col,
            sentence_input,
            tokenization,
        )

        st.dataframe(result_df)
        st.success(f"The '{sentence_input}' belong to class {max_prob_class}")
        st.balloons()
