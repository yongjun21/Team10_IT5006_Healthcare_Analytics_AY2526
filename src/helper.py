import streamlit as st
import pandas as pd

def render_navigation(previous_page, next_page):
    st.markdown("---")

    if previous_page is not None and next_page is not None:
        container = st.container(horizontal=True, horizontal_alignment="distribute")
        container.page_link(previous_page, label="Previous", icon=":material/chevron_left:")
        container.page_link(next_page, label="Next", icon=":material/chevron_right:")
        return
    
    if next_page is not None:
        container = st.container(horizontal=True, horizontal_alignment="right")
        container.page_link(next_page, label="Next", icon=":material/chevron_right:")
        return

    if previous_page is not None:
        st.page_link(previous_page, label="Previous", icon=":material/chevron_left:")
        return

def get_outcome_oh(dataset):
    return pd.get_dummies(dataset.data.targets, dtype=int)

def get_by_patient(data, outcome_oh):
    grouped_by_patient = data.groupby("patient_nbr")
    outcome_oh_grouped_by_patient = outcome_oh.groupby(data["patient_nbr"])
    return pd.DataFrame({
        "encounters": grouped_by_patient["encounter_id"].count(),
        "readmitted_<30": outcome_oh_grouped_by_patient["readmitted_<30"].mean(),
        "readmitted_>30": outcome_oh_grouped_by_patient["readmitted_>30"].mean(),
        "readmitted_NO": outcome_oh_grouped_by_patient["readmitted_NO"].mean(),
    })
