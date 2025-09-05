import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo

@st.cache_data
def load_data():
    return fetch_ucirepo(id=296)

if "dataset" not in st.session_state:
    dataset = load_data()
    data = dataset.data.original.copy()
    targets_oh = pd.get_dummies(dataset.data.targets, dtype=int)
    data = pd.merge(data, targets_oh, left_index=True, right_index=True)
    st.session_state.dataset = dataset
    st.session_state.data = data


landing_page = st.Page("landing.py", title="Start", default=True)
metadata_page = st.Page("metadata.py", title="Context", url_path="/context")
data_page = st.Page("data.py", title="Data", url_path="/data")
patient_page = st.Page("patient.py", title="Patient", url_path="/patient")

pg = st.navigation([landing_page, metadata_page, data_page, patient_page])
pg.run()
