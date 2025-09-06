import streamlit as st
from helper import render_navigation

st.title("Feature Analysis - Medications")

render_navigation("features/quantitative.py", "features/diagnoses.py")
