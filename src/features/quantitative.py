import streamlit as st
from helper import render_navigation

st.title("Feature Analysis - Quantitative")

render_navigation("features/admission.py", "features/medications.py")
