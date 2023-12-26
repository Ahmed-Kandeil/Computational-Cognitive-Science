import streamlit as st

st.set_page_config("CCS Project", page_icon="🧮")
st.title("About Us")

st.header("Personal Details")
st.write(
    """
    - **Name:** :orange[Ahmed Amro Abd-Elmoneim Moursi Kandil]
    - **ID:** :orange[4211133]
    - **Group:** :orange[A1]
    """
)

st.header("Educational Background")
st.write(
    "I am an ungraded University student currently pursuing a degree in :blue[***Delta University For Science And Technology***]. "
    "My academic journey has been focused on data science and artificial intelligence, "
    "where I've developed a passion for extracting insights from data."
)

st.header("Project Overview")
st.write(
    """
    The Computational Cognitive Science (CCS) project is a culmination of my interest 
    in data science and artificial intelligence. The project revolves around the implementation
    of various Naive Bayes classifiers, with a particular focus on solving equations in Natural
    Language Processing (NLP) and related fields.

    **Objectives:**
    - Implement and analyze the performance of different Naive Bayes classifiers.
    - Apply these classifiers to solve equations in NLP and other relevant domains.
    - Gain practical insights into the application of machine learning algorithms.

    **Expected Outcomes:**
    By the end of the project, I aim to become a proficient Data Scientist with a deeper
    understanding of the practical aspects of implementing machine learning models, especially
    Naive Bayes classifiers.
    """
)

st.header("Acknowledgments")
st.write(
    "I would like to express my gratitude to :red[***Dr. Sayed***] for providing guidance "
    "and mentorship throughout this project. Additionally, I appreciate the support and resources "
    "provided by :blue[***Delta University For Science And Technology***] that have made this "
    "research endeavor possible."
)
