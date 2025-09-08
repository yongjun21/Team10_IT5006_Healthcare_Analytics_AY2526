import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo

from helper import get_outcome_oh

@st.cache_data
def load_data():
    return fetch_ucirepo(id=296)

if "dataset" not in st.session_state:
    dataset = load_data()
    data = dataset.data.original.copy()
    data["readmitted"] = data["readmitted"].astype('category')
    data["readmitted"] = data["readmitted"].cat.reorder_categories(['<30', '>30', 'NO'])
    outcome_oh = get_outcome_oh(dataset)
    st.session_state.dataset = dataset
    st.session_state.data = data
    st.session_state.outcome_oh = outcome_oh

pg = st.navigation([
    st.Page("landing.py", title="Start", default=True),
    st.Page("metadata.py", title="Context", url_path="/context"),
    st.Page("data.py", title="Data", url_path="/data"),
    st.Page("patient.py", title="Patient", url_path="/patient"),
    st.Page("features/index.py", title="Features", url_path="/features"),
    st.Page("features/demographics.py", title="Demographics", url_path="/demographics", icon=":material/line_start:"),
    st.Page("features/admission.py", title="Admission Type / Discharge Disposition", url_path="/admission", icon=":material/line_start:"),
    st.Page("features/quantitative.py", title="Quantitative Features", url_path="/quantitatives", icon=":material/line_start:"),
    st.Page("features/medications.py", title="Medications", url_path="/medications", icon=":material/line_start:"),
    st.Page("features/diagnoses.py", title="Diagnoses", url_path="/diagnoses", icon=":material/line_start:"),
    st.Page("features/lab.py", title="Lab Results", url_path="/lab", icon=":material/line_start:"),
    st.Page("mixed.py", title="Mixed Model Analysis", url_path="/mixed_model"),
])
pg.run()
