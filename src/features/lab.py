import streamlit as st
from helper import render_navigation

st.title("Feature Analysis - Lab Results")

render_navigation("features/diagnoses.py", "mixed.py")
