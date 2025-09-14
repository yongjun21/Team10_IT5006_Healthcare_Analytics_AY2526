import streamlit as st
import pandas as pd
import numpy as np

def convert_to_category(data, columns):
    for column in columns:
        data[column] = data[column].astype('category')
    return data

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

@st.cache_data
def get_outcome_by_feature(data, weighted_outcome_oh, feature, sorted=False, patient_weighted=False):
    # Group by feature and sum the weighted one-hot columns
    outcome_by_feature = weighted_outcome_oh.groupby(data[feature]).sum(
    ) if patient_weighted else data.groupby(feature)['readmitted'].value_counts().unstack(fill_value=0)

    if not sorted:
        return outcome_by_feature

    # Sort by total counts if requested
    counts = data[feature].value_counts().sort_values(ascending=False)
    return outcome_by_feature.reindex(counts.index)

@st.cache_data
def get_scatter_data(data, x_features, y_features):
    x_jitter = np.random.uniform(-0.5, 0.5, len(data))
    y_jitter = np.random.uniform(-0.5, 0.5, len(data))
    return data[x_features].sum(axis=1) + x_jitter, data[y_features].sum(axis=1) + y_jitter, 
