import streamlit as st
import pandas as pd
from scipy.stats import chi2_contingency
import altair as alt
import json

from helper import render_navigation, get_outcome_by_feature

data = st.session_state.data

with open('src/assets/metadata.json', 'r') as f:
    metadata = json.load(f)
    
labels = {
    "admission_type_id": {item['cat']: item['label'] for item in metadata['admission_type_labels']},
    "admission_source_id": {item['cat']: item['label'] for item in metadata['admission_source_labels']},
    "discharge_disposition_id": {item['cat']: item['label'] for item in metadata['discharge_disposition_labels']}
}

data["admission_type_id"] = data["admission_type_id"].astype('category')
data["admission_source_id"] = data["admission_source_id"].astype('category')
data["discharge_disposition_id"] = data["discharge_disposition_id"].astype(
    'category')

def format_feature_name(feature):
    return feature[:-3].replace("_", " ").title()

@st.cache_data
def get_chart_data(outcome_by_feature, selected_feature, exclude_categories = []):
    feature_order = outcome_by_feature.index.tolist()

    chart_data = outcome_by_feature.reset_index().melt(
        id_vars=[selected_feature],
        var_name='readmitted',
        value_name='count'
    )
    
    # Clean up the outcome column names
    chart_data['readmitted'] = chart_data['readmitted'].str.replace('readmitted_', '')

    # Exclude Unknown/Invalid for gender
    if selected_feature == "admission_type_id":
        chart_data = chart_data[chart_data["admission_type_id"].isin([1, 2, 3])]
    else:
        count_of_outcome = outcome_by_feature.sum(axis=1)
        groups_to_keep = count_of_outcome[count_of_outcome / count_of_outcome.sum() >= 0.03]
        groups_to_keep = groups_to_keep[groups_to_keep.index.map(labels[selected_feature]) != "NULL"]
        chart_data = chart_data[chart_data[selected_feature].isin(groups_to_keep.index)]
    
    return chart_data, feature_order


st.title("Feature Analysis - Admission Type, Admitting Source & Discharge Disposition")

st.subheader("Distribution of Readmitted by Feature")

selected_feature = st.selectbox(
    "Feature", ["admission_type_id", "admission_source_id", "discharge_disposition_id"], format_func=format_feature_name)
chart_type = st.selectbox(
    "View As", ["Count", "Proportion"])

outcome_by_feature = get_outcome_by_feature(
    data, None, selected_feature
)

chart_data, feature_order = get_chart_data(
    outcome_by_feature, selected_feature)

# Add formatted labels to chart data
chart_data['formatted_label'] = chart_data[selected_feature].map(labels[selected_feature])

if chart_type == "Count":    

    
    # Create grouped bar chart
    base = alt.Chart(chart_data).encode(
        x=alt.X('formatted_label:O',
                title=format_feature_name(selected_feature),
                axis=alt.Axis(labelAngle=-90 if selected_feature == "discharge_disposition_id" else -45, labelLimit=300),
                sort=feature_order),  # Explicitly specify the sort order
        y=alt.Y('count:Q', title='Count'),
        xOffset='readmitted:N',
        color=alt.Color('readmitted:N',
                        scale=alt.Scale(domain=['<30', '>30', 'NO'],
                                        range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                        legend=alt.Legend(title='Readmission Status'))
    ).properties(
        height=700 if selected_feature == "discharge_disposition_id" else 500
    )

    st.markdown(f'##### Readmitted Distribution by {format_feature_name(selected_feature)}')
    bars = base.mark_bar()
    st.altair_chart(bars)

elif chart_type == "Proportion":
    # Create horizontal stacked bar chart
    base_pct = alt.Chart(chart_data).encode(
        x=alt.X('count:Q', title='Percentage (%)',
                stack="normalize",
                axis=alt.Axis(format='%'),
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('formatted_label:O',
                axis=alt.Axis(labelLimit=300),
                title=format_feature_name(selected_feature),
                sort=feature_order),
        color=alt.Color('readmitted:N',
                        scale=alt.Scale(domain=['<30', '>30', 'NO'],
                                        range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                        legend=alt.Legend(title='Readmission Status'))
    ).properties(
        height=250
    )

    st.markdown(f'##### Readmitted Distribution by {format_feature_name(selected_feature)} (%)')
    bars_pct = base_pct.mark_bar()
    st.altair_chart(bars_pct)

render_navigation("features/demographics.py", "features/quantitative.py")
