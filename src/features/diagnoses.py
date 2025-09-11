import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from helper import render_navigation

data = st.session_state.data

# Combine all unique diagnosis values from diag_1, diag_2, and diag_3
all_diag_categories = pd.concat([data["diag_1"], data["diag_2"], data["diag_3"]]).unique()
# Remove None/NaN values from categories
all_diag_categories = all_diag_categories[pd.notna(all_diag_categories)]
data["diag_1"] = pd.Categorical(data["diag_1"], categories=all_diag_categories)
data["diag_2"] = pd.Categorical(data["diag_2"], categories=all_diag_categories)
data["diag_3"] = pd.Categorical(data["diag_3"], categories=all_diag_categories)



# Perform PCA on combined diagnosis one-hot encodings
@st.cache_data
def get_pca(data):
    diag_1_oh = pd.get_dummies(data["diag_1"], dtype=int)
    diag_2_oh = pd.get_dummies(data["diag_2"], dtype=int)
    diag_3_oh = pd.get_dummies(data["diag_3"], dtype=int)

    # Combine diagnosis one-hot encodings with OR operation
    diag_oh_combined = diag_1_oh | diag_2_oh | diag_3_oh
    
    # Get the most frequent diagnosis
    most_freq_diag = diag_oh_combined.sum().sort_values(ascending=False).index


    # Create dataframe with patient numbers and diagnosis one-hot encodings
    patient_diag_oh = pd.concat([data['patient_nbr'], diag_oh_combined], axis=1)

    # Group by patient and combine using max (equivalent to OR for binary values)
    patient_grouped_diag_oh = patient_diag_oh.groupby('patient_nbr').max()

    pca = PCA()
    pca_result = pca.fit_transform(patient_grouped_diag_oh[most_freq_diag])
    
    return pca, pca_result



#region
# Get top 10 diagnoses in the dataset
st.title("Top 10 Diagnoses in Dataset")
all_diagnosis = pd.concat([data["diag_1"], data["diag_2"], data["diag_3"]]).value_counts()
top10_diagnosis = all_diagnosis.head(10)

# Creating a sub map of ICD-9 codes 
icd9_sub_dict = {
    "250": "Diabetes Mellitus",
    "276": "Fluid and electrolyte disorders",
    "414": "Other forms of chronic ischemic heart disease",
    "401": "Essential hypertension",
    "427": "Cardiac dysrhythmias",
    "428": "Heart failure",
    "496": "Chronic airway obstruction",
    "403": "Hypertensive chronic kidney disease",
    "486": "Pneumonia, organism unspecified",
    "786": "Respiratory symptoms",
    "599": "Urinary tract infection",
    "272": "Disorders of lipoid metabolism"
}

x_labels = [icd9_sub_dict.get(code) for code in top10_diagnosis.index]
# Plot with Matplotlib
fig, ax = plt.subplots(figsize=(8,5))
top10_diagnosis.sort_values(ascending=False).plot(kind="bar", ax=ax)
ax.set_title("Top 10 Diagnosis Counts")
ax.set_xlabel("Diagnosis Code")
ax.set_ylabel("Count")

# Update x-axis tick labels to show descriptions instead of codes
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=45, ha="right")
#plt.xticks(rotation=45)

# Show in Streamlit
st.pyplot(fig)
#endregion


st.title("Feature Analysis - Diagnoses")

pca, pca_result = get_pca(data)

# Calculate explained variance ratio
explained_var_ratio = pca.explained_variance_ratio_


# Prepare data for Altair
cum_explained_var = np.cumsum(explained_var_ratio[:100])
altair_df = pd.DataFrame({
    'Number of Components': np.arange(1, 101),
    'Cumulative Explained Variance Ratio': cum_explained_var
})

# Altair line chart
chart = alt.Chart(altair_df).mark_line(point=True).encode(
    x=alt.X('Number of Components', title='Number of Components'),
    y=alt.Y('Cumulative Explained Variance Ratio', title='Cumulative Explained Variance Ratio')
).properties(
    title='PCA on Combined Diagnosis Features',
    width=600,
    height=400
).interactive()

st.altair_chart(chart, use_container_width=True)

