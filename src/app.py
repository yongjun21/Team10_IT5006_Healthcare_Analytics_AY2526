import streamlit as st
import pandas as pd
import pickle

# from ucimlrepo import fetch_ucirepo

from helper import get_outcome_oh, convert_to_category


@st.cache_data
def load_data():
    # return fetch_ucirepo(id=296)
    with open('src/assets/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    return dataset


if "dataset" not in st.session_state:
    dataset = load_data()
    data = dataset.data.original
    convert_to_category(data, [
        "readmitted",
        "race", "gender", "age", "weight",
        "payer_code", "medical_specialty",
        "admission_type_id", "admission_source_id", "discharge_disposition_id",
        "max_glu_serum", "A1Cresult",
    ])

    medication_cols = dataset.variables[dataset.variables["description"].str.startswith("The feature indicates whether the drug was prescribed")]["name"]
    for col in medication_cols:
        data[col] = data[col].astype('category')

    # Combine all unique diagnosis values from diag_1, diag_2, and diag_3
    all_diag_categories = pd.concat(
        [data["diag_1"], data["diag_2"], data["diag_3"]]).unique()
    # Remove None/NaN values from categories
    all_diag_categories = all_diag_categories[pd.notna(all_diag_categories)]
    data["diag_1"] = pd.Categorical(data["diag_1"], categories=all_diag_categories)
    data["diag_2"] = pd.Categorical(data["diag_2"], categories=all_diag_categories)
    data["diag_3"] = pd.Categorical(data["diag_3"], categories=all_diag_categories)

    data.drop(["change", "diabetesMed"], axis=1, inplace=True)
    
    data["readmitted"] = data["readmitted"].cat.reorder_categories(
        ['<30', '>30', 'NO'])
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
    st.Page("features/demographics.py", title="Demographics",
            url_path="/demographics", icon=":material/line_start:"),
    st.Page("features/admission.py", title="Admission Type / Discharge Disposition",
            url_path="/admission", icon=":material/line_start:"),
    st.Page("features/numerical.py", title="Numerical Features",
            url_path="/numerical", icon=":material/line_start:"),
    st.Page("features/medications.py", title="Medications",
            url_path="/medications", icon=":material/line_start:"),
    st.Page("features/diagnoses.py", title="Diagnoses",
            url_path="/diagnoses", icon=":material/line_start:"),
    st.Page("features/lab.py", title="Lab Results",
            url_path="/lab", icon=":material/line_start:"),
    st.Page("mixed.py", title="Mixed Model Analysis", url_path="/mixed_model"),
])
pg.run()
