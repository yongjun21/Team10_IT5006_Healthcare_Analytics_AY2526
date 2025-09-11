import streamlit as st
from helper import render_navigation

st.title("Feature Analysis")

st.page_link("features/demographics.py", label="Demographics", icon=":material/arrow_outward:")
st.page_link("features/admission.py", label="Admission Type / Discharge Disposition", icon=":material/arrow_outward:")
st.page_link("features/numerical.py", label="Numerical Features", icon=":material/arrow_outward:")
st.page_link("features/medications.py", label="Medications", icon=":material/arrow_outward:")
st.page_link("features/diagnoses.py", label="Diagnoses", icon=":material/arrow_outward:")
st.page_link("features/lab.py", label="Lab Results", icon=":material/arrow_outward:")

render_navigation("patient.py", "features/demographics.py")