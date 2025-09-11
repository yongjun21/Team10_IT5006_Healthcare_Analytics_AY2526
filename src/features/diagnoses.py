import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import json

from sklearn.decomposition import PCA

from helper import render_navigation

data = st.session_state.data

# Combine all unique diagnosis values from diag_1, diag_2, and diag_3
all_diag_categories = pd.concat(
    [data["diag_1"], data["diag_2"], data["diag_3"]]).unique()
# Remove None/NaN values from categories
all_diag_categories = all_diag_categories[pd.notna(all_diag_categories)]
data["diag_1"] = pd.Categorical(data["diag_1"], categories=all_diag_categories)
data["diag_2"] = pd.Categorical(data["diag_2"], categories=all_diag_categories)
data["diag_3"] = pd.Categorical(data["diag_3"], categories=all_diag_categories)

diag_value_counts = pd.concat(
    [data["diag_1"], data["diag_2"], data["diag_3"]]).value_counts()

with open('src/assets/metadata.json', 'r') as f:
    metadata = json.load(f)

diagnosis_labels = {item["cat"]: item["label"]
                    for item in metadata["diagnosis_labels"]}

icd9_labels = pd.read_csv('src/assets/ICP9.csv', index_col='code')['diag']


# Perform PCA on combined diagnosis binary encodings
@st.cache_data
def get_pca(data):
    diag_1_oh = pd.get_dummies(data["diag_1"], dtype=int)
    diag_2_oh = pd.get_dummies(data["diag_2"], dtype=int)
    diag_3_oh = pd.get_dummies(data["diag_3"], dtype=int)

    # Combine diagnosis one-hot encodings with OR operation
    diag_oh_combined = diag_1_oh | diag_2_oh | diag_3_oh

    # Create dataframe with patient numbers and diagnosis one-hot encodings
    patient_diag_oh = pd.concat(
        [data['patient_nbr'], diag_oh_combined], axis=1)

    # Group by patient and combine using max (equivalent to OR for binary values)
    patient_grouped_diag_oh = patient_diag_oh.groupby('patient_nbr').max()

    pca = PCA()
    pca_result = pca.fit_transform(patient_grouped_diag_oh[diag_value_counts.index])

    return pca, pca_result


# region
st.title("Feature Analysis - Diagnoses")

st.subheader("Top 10 Diagnoses")

top10_diagnosis = diag_value_counts.head(10)
chart_data = pd.DataFrame({
    'Diagnosis': top10_diagnosis.index.map(diagnosis_labels),
    'Prevalence': top10_diagnosis / len(data) * 100
})

chart = alt.Chart(chart_data).mark_bar().encode(
    y=alt.Y('Diagnosis', sort='-x', axis=alt.Axis(labelLimit=300)),
    x=alt.X('Prevalence', title='Prevalence (%)'),
    tooltip=['Diagnosis', 'Prevalence']
).properties(
    width=600,
    height=400
)

st.altair_chart(chart, use_container_width=True)


st.subheader("PCA on Diagnosis Features")

pca, pca_result = get_pca(data)

cum_explained_var = np.cumsum(pca.explained_variance_ratio_)
altair_df = pd.DataFrame({
    'Number of Components': np.arange(1, len(pca.explained_variance_ratio_) + 1),
    'Cumulative Explained Variance Ratio': cum_explained_var
})

chart = alt.Chart(altair_df).mark_line(point=True).encode(
    x=alt.X('Number of Components', title='Number of Components', scale=alt.Scale(domain=[0, 200])),
    y=alt.Y('Cumulative Explained Variance Ratio',
            title='Cumulative Explained Variance Ratio')
).properties(
    title='PCA on Combined Diagnosis Features',
    width=600,
    height=400
).interactive(bind_y=False)

st.altair_chart(chart, use_container_width=True)


st.subheader("PCA Components Composition")

# Create DataFrame of first 30 PCA components
diag_pca_components = pd.DataFrame(
    pca.components_[:30, :30],
    columns=diag_value_counts.index[:30],
    index=[f'PC{i:02d}' for i in range(1, 31)]
)
# Map diagnosis codes to labels, padding with zeros
diag_pca_components.columns = [f'{code} - {icd9_labels.get(code.zfill(3), "")}' for code in diag_pca_components.columns]

# Convert DataFrame to long format for Altair
heatmap_data = diag_pca_components.reset_index().melt(
    id_vars=['index'],
    var_name='Diagnosis',
    value_name='Value'
)

# Create heatmap using Altair
heatmap = alt.Chart(heatmap_data).mark_rect().encode(
    x=alt.X('Diagnosis:N', 
            scale=alt.Scale(domain=list(diag_pca_components.columns)),
            axis=alt.Axis(labelAngle=-90, labelLimit=300),
            title='Diagnosis'),
    y=alt.Y('index:N', title='PCA Components'),
    color=alt.Color('Value:Q',
                   scale=alt.Scale(domain=[-1, 1], range=['#2166ac', '#f7f7f7', '#b2182b']),
                   legend=None),
    tooltip=['index', 'Diagnosis', 'Value']
).properties(
    width=800,
    height=800
)

text = heatmap.mark_text(baseline='middle').encode(
    text=alt.Text('Value:Q', format='.2f'),
    color=alt.condition(
        abs(alt.datum.Value) > 0.5,
        alt.value('white'),
        alt.value('black')
    )
)

st.markdown("""
<style>
canvas[style*="width: 800px"] {
    margin-left: -70px;
}
""", unsafe_allow_html=True)

st.altair_chart(heatmap + text, use_container_width=False, key="heatmap")
