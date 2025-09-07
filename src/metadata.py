import streamlit as st
import pandas as pd
from helper import render_navigation

st.title("Context")

dataset = st.session_state.dataset

st.subheader("Abstract")
st.text(dataset.metadata.abstract)

st.subheader("Repository URL")
st.markdown(f"[{dataset.metadata.repository_url}]({dataset.metadata.repository_url})")

st.subheader("Paper Information")
paper_info = pd.DataFrame({
    'Key': ['Title', 'Authors', 'Year', 'Journal', 'URL'],
    'Value': [
        dataset.metadata.intro_paper['title'],
        dataset.metadata.intro_paper['authors'],
        dataset.metadata.intro_paper['year'],
        dataset.metadata.intro_paper['venue'],
        dataset.metadata.intro_paper['URL'],
    ]
})
st.dataframe(
    paper_info,
    hide_index=True
)

st.subheader("Variables")
st.dataframe(dataset.variables)

render_navigation("landing.py", "data.py")
