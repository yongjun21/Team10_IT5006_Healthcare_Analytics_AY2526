import streamlit as st
from helper import render_navigation

st.title("Feature Analysis - Admission Type / Discharge Disposition")

render_navigation("features/demographics.py", "features/quantitative.py")
