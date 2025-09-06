import streamlit as st
from helper import render_navigation

st.title("Feature Analysis - Diagnoses")

render_navigation("features/medications.py", "features/lab.py")
