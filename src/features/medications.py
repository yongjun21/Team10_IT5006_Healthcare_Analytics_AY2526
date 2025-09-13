import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from helper import render_navigation

dataset = st.session_state.dataset
data = st.session_state.data
outcome_oh = st.session_state.outcome_oh

if "medication_bool" not in st.session_state:
    medication_cols = dataset.variables[dataset.variables["description"].str.startswith("The feature indicates whether the drug was prescribed")]["name"]
    st.session_state.medication_bool = data[medication_cols].apply(lambda x: (x != 'No').astype(int))
medication_bool = st.session_state.medication_bool

med_usage = pd.DataFrame({
    "<30": (medication_bool.mul(outcome_oh["readmitted_<30"], axis=0)).sum(axis=0) / len(data) * 100,
    ">30": (medication_bool.mul(outcome_oh["readmitted_>30"], axis=0)).sum(axis=0) / len(data) * 100,
    "NO": (medication_bool.mul(outcome_oh["readmitted_NO"], axis=0)).sum(axis=0) / len(data) * 100
})

med_usage = med_usage.reindex(med_usage.sum(axis=1).sort_values(ascending=False).index)

@st.cache_data
def get_pca(medication_bool):
    pca = PCA()
    pca.fit_transform(medication_bool)
    return pca


st.title("Feature Analysis - Medications")


st.subheader("Usage")

# Convert to long format for Altair
chart_data = med_usage.reset_index().melt(
    id_vars='index',
    var_name='readmission_status',
    value_name='percentage'
)

# Create Altair stacked bar chart
chart = alt.Chart(chart_data).mark_bar().encode(
    x=alt.X('index:O', 
            title='Medication',
            axis=alt.Axis(labelAngle=-90, labelLimit=100),
            sort='-y'),
    y=alt.Y('percentage:Q', 
            title='Percentage of Cases (%)',
            stack='zero'),
    color=alt.Color('readmission_status:N',
                    scale=alt.Scale(domain=['<30', '>30', 'NO'],
                                    range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                    legend=alt.Legend(title='Readmission Status',
                                    orient='top-right',
                                    fillColor="rgba(255, 255, 255, 0.7)")),
    tooltip=['index', 'readmission_status', alt.Tooltip('percentage:Q', format='.2f')]
).properties(
    height=600
)

st.markdown(f'##### Medication Usage Rates And Readmission Outcome')
st.altair_chart(chart)

st.markdown("**Observation** The most common medication prescribed is insulin (> 50% of the time) which isn't surprising for a study focused on diabetic patients. The usage chart has a long tail: most usage concentrated in the top 10 prescribed medications while the rest appears < 1% of the time.")


st.subheader("Outcome Comparison")

# Get top 10 medications by total usage (descending order)
top10_medications = med_usage.sum(axis=1).sort_values(ascending=False).head(10).index.tolist()

# Filter chart_data to only include top 10 medications
filtered_chart_data = chart_data[chart_data['index'].isin(top10_medications)]

# Create Altair horizontal stacked bar chart with internal normalization
norm_chart = alt.Chart(filtered_chart_data).mark_bar().encode(
    x=alt.X('percentage:Q', 
            title='Proportion of Cases',
            stack='normalize',
            axis=alt.Axis(format='%')),
    y=alt.Y('index:O', 
            title='Medication',
            scale=alt.Scale(domain=top10_medications),
            sort=None),
    color=alt.Color('readmission_status:N',
                    scale=alt.Scale(domain=['<30', '>30', 'NO'],
                                    range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                    legend=alt.Legend(title='Readmission Status',
                                    orient='top-right',
                                    fillColor="rgba(255, 255, 255, 0.7)")),
    tooltip=['index', 'readmission_status', alt.Tooltip('percentage:Q', format='.2f')]
).properties(
    height=400
)

st.markdown(f'##### Distribution of Readmission Outcome when Medication is Prescribed')
st.altair_chart(norm_chart, use_container_width=True)


st.subheader("Correlation Analysis")

# Create correlation matrix
medication_corr_data = pd.concat([
    outcome_oh["readmitted_<30"],
    data["time_in_hospital"],
    medication_bool[med_usage.index[:10]]
], axis=1)
corr_matrix = medication_corr_data.corr()

# Convert correlation matrix to long format for Altair
corr_data = corr_matrix.reset_index().melt(
    id_vars='index',
    var_name='variable2',
    value_name='correlation'
)

# Create mask to exclude diagonal (self-correlations)
corr_data = corr_data[corr_data['index'] != corr_data['variable2']]

# Get the original column order from medication_corr_data
column_order = medication_corr_data.columns.tolist()

# Create Altair heatmap
heatmap = alt.Chart(corr_data).mark_rect().encode(
    x=alt.X('variable2:O', 
            title='Variables',
            axis=alt.Axis(labelAngle=-45, labelLimit=100),
            scale=alt.Scale(domain=column_order)),
    y=alt.Y('index:O', 
            title='Variables',
            axis=alt.Axis(labelLimit=100),
            scale=alt.Scale(domain=column_order)),
    color=alt.Color('correlation:Q',
                    scale=alt.Scale(domain=[-0.2, 0.2], 
                                    range=['#2166ac', '#f7f7f7', '#b2182b']),
                    legend=alt.Legend(title='Correlation')),
    tooltip=['index', 'variable2', alt.Tooltip('correlation:Q', format='.3f')]
).properties(
    width=600,
    height=500
)

# Add text annotations
text = heatmap.mark_text(baseline='middle').encode(
    text=alt.Text('correlation:Q', format='.2f'),
    color=alt.condition(
        abs(alt.datum.correlation) < 0.1,
        alt.value('black'),
        alt.value('white')
    )
)

st.markdown('##### Medication Co-occurrence Correlation Matrix')
st.text("Includes outcome variables")
correlation_chart = (heatmap + text).resolve_scale(color='independent')
st.altair_chart(correlation_chart, use_container_width=True)


st.subheader("PCA Analysis")

pca = get_pca(medication_bool)

# Create PCA explained variance data
pca_data = pd.DataFrame({
    'components': range(1, len(pca.explained_variance_ratio_) + 1),
    'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
    'individual_variance': pca.explained_variance_ratio_
})

# Create Altair line chart for cumulative explained variance
pca_chart = alt.Chart(pca_data).mark_line(point=True, strokeWidth=3).encode(
    x=alt.X('components:O', 
            title='Number of Components',
            axis=alt.Axis(grid=True)),
    y=alt.Y('cumulative_variance:Q', 
            title='Cumulative Explained Variance Ratio',
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format='%')),
    color=alt.value('#1f77b4'),
    tooltip=['components', alt.Tooltip('cumulative_variance:Q', format='.3f')]
).properties(
    width=600,
    height=400
)

st.markdown('##### PCA Explained Variance Ratio')
st.altair_chart(pca_chart, use_container_width=True)

# Create a DataFrame with PCA components
pca_components = pd.DataFrame(
    pca.components_[:10, medication_bool.columns.get_indexer(med_usage.index[:10])],
    columns=med_usage.index[:10],
    index=[f'PC{i:02d}' for i in range(1, 11)]
)

# Convert PCA components to long format for Altair
pca_components_long = pca_components.reset_index().melt(
    id_vars='index',
    var_name='medication',
    value_name='component_value'
)

# Create Altair heatmap for PCA components
pca_heatmap = alt.Chart(pca_components_long).mark_rect().encode(
    x=alt.X('medication:O', 
            title='Medications',
            scale=alt.Scale(domain=med_usage.index[:10]),
            axis=alt.Axis(labelAngle=-90, labelLimit=100)),
    y=alt.Y('index:O', 
            title='PCA Components',
            axis=alt.Axis(labelLimit=100)),
    color=alt.Color('component_value:Q',
                    scale=alt.Scale(domain=[-1, 1], 
                                    range=['#2166ac', '#f7f7f7', '#b2182b']),
                    legend=alt.Legend(title='Component Value')),
    tooltip=['index', 'medication', alt.Tooltip('component_value:Q', format='.3f')]
).properties(
    title='First 10 PCA Components Composition',
    width=600,
    height=600
)

# Add text annotations
pca_text = pca_heatmap.mark_text(baseline='middle').encode(
    text=alt.Text('component_value:Q', format='.2f'),
    color=alt.condition(
        abs(alt.datum.component_value) < 0.1,
        alt.value('black'),
        alt.value('white')
    )
)

# Combine heatmap and text
pca_components_chart = (pca_heatmap + pca_text).resolve_scale(color='independent')

st.altair_chart(pca_components_chart, use_container_width=True)

st.markdown("**Observation**")
st.markdown("The component composition chart shows some interesting patterns that confirm some of our earlier observations on the correlation heatmap. The first principal component (PC) is almost exclusively the effect of insulin showing insulin is indeed often used independently. The clustering of composition weights for PC2, PC3 and PC4 shows two distinct usage patterns: 1. using glipizide but not glyburide and 2. using glipizide together with glyburide. Similar patterns can be observed in the PC5 & PC6 cluster. Thus PCA proves to be useful in identifying usage patterns even in the absence of domain knowledge.")
st.markdown("The concentration of composition weights along the diagonal with medications sorted by usage suggests the dominance of prevalence effect. But some clustering can still be observed. The dominance of insulin usage and long tail effect explains why just the first 7 principal components cover > 95% of data variance.")

render_navigation("features/numerical.py", "features/diagnoses.py")
