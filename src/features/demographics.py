import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

from helper import render_navigation, get_by_patient

data = st.session_state.data
outcome_oh = st.session_state.outcome_oh

data["race"] = data["race"].astype('category')
data["gender"] = data["gender"].astype('category')
data["age"] = data["age"].astype('category')

if "by_patient" not in st.session_state:
    st.session_state.by_patient = get_by_patient(data, outcome_oh)
by_patient = st.session_state.by_patient

demographic_weight = 1.0 / data["patient_nbr"].map(by_patient["encounters"])
weighted_outcome_oh = outcome_oh.multiply(demographic_weight, axis=0)


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


st.title("Feature Analysis - Demographics")

st.subheader("Distribution of Demographics")

selected_feature = st.selectbox(
    "Feature", ["race", "gender", "age"], format_func=lambda x: x.title())
chart_type = st.selectbox(
    "View As", ["Count", "Proportion", "Mosaic", "Mosaic With Pearson Residuals"])

outcome_by_feature = get_outcome_by_feature(
    data, weighted_outcome_oh, selected_feature, sorted=selected_feature != "age", patient_weighted=True
)


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
    if selected_feature == "gender":
        chart_data = chart_data[~chart_data["gender"].isin(exclude_categories)]
        feature_order = [f for f in feature_order if f not in exclude_categories]
    
    return chart_data, feature_order

chart_data, feature_order = get_chart_data(outcome_by_feature, selected_feature, ["Unknown/Invalid"] if selected_feature == "gender" else [])

if chart_type == "Count":
    base = alt.Chart(chart_data).encode(
        x=alt.X(f'{selected_feature}:O',
                title=selected_feature.title(),
                axis=alt.Axis(labelAngle=-45),
                sort=feature_order),  # Explicitly specify the sort order
        y=alt.Y('count:Q', title='Count'),
        color=alt.Color('readmitted:N',
                        scale=alt.Scale(domain=['<30', '>30', 'NO'],
                                        range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                        legend=alt.Legend(title='Readmission Status'))
    ).properties(
        title=f"Readmitted Distribution by {selected_feature.title()}",
        height=600
    )

    bars = base.mark_bar(size=50)
    st.altair_chart(bars)
elif chart_type == "Proportion":
    # Create horizontal stacked bar chart
    base_pct = alt.Chart(chart_data).encode(
        x=alt.X('count:Q', title='Percentage (%)',
                stack="normalize",
                axis=alt.Axis(format='%'),
                scale=alt.Scale(domain=[0, 1])),
        y=alt.Y(f'{selected_feature}:O',
                title=selected_feature.title(),
                sort=feature_order),
        color=alt.Color('readmitted:N',
                        scale=alt.Scale(domain=['<30', '>30', 'NO'],
                                        range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                        legend=alt.Legend(title='Readmission Status'))
    ).properties(
        title=f"Readmitted Distribution by {selected_feature.title()} (%)",
        height=400
    )

    bars_pct = base_pct.mark_bar()
    st.altair_chart(bars_pct)

elif chart_type == "Mosaic":
    demographic_total_counts = chart_data.groupby(selected_feature)['count'].sum()
    outcome_total_counts = chart_data.groupby('readmitted')['count'].sum()
    total_population = demographic_total_counts.sum()
    demographic_proportions = demographic_total_counts / total_population
    outcome_proportions = outcome_total_counts / total_population

    count = {}
    for _, row in chart_data.iterrows():
        count[(row[selected_feature], row['readmitted'])] = row['count']


    mosaic_data = []
    label_data = []

    gap = 0.002

    cum_y = 0
    for feature in feature_order:
        cum_x = 0
        height = demographic_proportions[feature]
        
        for readmitted in ['<30', '>30', 'NO']:
            width = count[feature, readmitted] / demographic_total_counts[feature]
            mosaic_data.append({
                "x": cum_x + 0.5 * gap,
                "y": 1 - (cum_y + 0.5 * gap),
                "x2": cum_x + width - 0.5 * gap,
                "y2": 1 - (cum_y + height - 0.5 * gap),
                "color": readmitted,
            })
            cum_x += width

        label_data.append({
            'feature': feature,
            'y_mid': 1 - (cum_y + height / 2),
            'x_mid': 0
        })

        cum_y += height
    
    mosaic_df = pd.DataFrame(mosaic_data)    
    label_df = pd.DataFrame(label_data)
    
    base_mosaic = alt.Chart(mosaic_df).encode(
        x=alt.X('x:Q', 
                title='Readmission Proportion',
                axis=alt.Axis(format='%'),
                scale=alt.Scale(domain=[0, 1])),
        x2='x2:Q',
        y=alt.Y('y:Q', 
                title=selected_feature.title(),
                axis=alt.Axis(format='%'),
                scale=alt.Scale(domain=[0, 1])),
        y2='y2:Q',
        color=alt.Color('color:O',
                        scale=alt.Scale(domain=['<30', '>30', 'NO'],
                                        range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                        legend=alt.Legend(title='Readmission Status',
                                        orient='top-right',
                                        fillColor="rgba(255, 255, 255, 0.7)"))
    ).properties(
        title=f"Mosaic Plot: {selected_feature.title()} Distribution vs Readmission Status",
        height=600,
    )
    
    # Create text labels for demographic values
    text_labels = alt.Chart(label_df).mark_text(
        align='left',
        dx=5,
        fontSize=12,
    ).encode(
        x=alt.X('x_mid:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('y_mid:Q', scale=alt.Scale(domain=[0, 1])),
        text='feature:N'
    )
    
    # Combine mosaic plot with text labels
    mosaic_plot = base_mosaic.mark_rect() + text_labels
    st.altair_chart(mosaic_plot)

elif chart_type == "Mosaic With Pearson Residuals":
    demographic_total_counts = chart_data.groupby(selected_feature)['count'].sum()
    outcome_total_counts = chart_data.groupby('readmitted')['count'].sum()
    total_population = demographic_total_counts.sum()
    demographic_proportions = demographic_total_counts / total_population
    outcome_proportions = outcome_total_counts / total_population

    count = {}
    for _, row in chart_data.iterrows():
        count[(row[selected_feature], row['readmitted'])] = row['count']

    mosaic_data = []
    label_data = []

    gap = 0.002

    cum_y = 0
    for feature in feature_order:
        cum_x = 0
        height = demographic_proportions[feature]
        
        for readmitted in ['<30', '>30', 'NO']:
            # Get observed count
            observed = count[(feature, readmitted)]
            
            # Calculate expected count
            feature_total = demographic_total_counts[feature]
            outcome_total = outcome_total_counts[readmitted]
            expected = (feature_total * outcome_total) / total_population
            
            # Compute Pearson residual
            if expected > 0:
                residual = (observed - expected) / np.sqrt(expected)
            else:
                residual = 0
            
            # Calculate width for this segment
            width = observed / feature_total
            
            mosaic_data.append({
                "x": cum_x + 0.5 * gap,
                "y": 1 - (cum_y + 0.5 * gap),
                "x2": cum_x + width - 0.5 * gap,
                "y2": 1 - (cum_y + height - 0.5 * gap),
                "color": readmitted,
                "residual": residual,
                "observed": observed,
                "expected": expected
            })
            cum_x += width

        label_data.append({
            'feature': feature,
            'y_mid': 1 - (cum_y + height / 2),
            'x_mid': 0
        })

        cum_y += height
    
    # Get max absolute residual for color normalization
    max_abs_residual = max(abs(row['residual']) for row in mosaic_data)

    mosaic_df = pd.DataFrame(mosaic_data)    
    label_df = pd.DataFrame(label_data)
    
    base_mosaic = alt.Chart(mosaic_df).encode(
        x=alt.X('x:Q', 
                title='Readmission Proportion',
                axis=alt.Axis(format='%'),
                scale=alt.Scale(domain=[0, 1])),
        x2='x2:Q',
        y=alt.Y('y:Q', 
                title=selected_feature.title(),
                axis=alt.Axis(format='%'),
                scale=alt.Scale(domain=[0, 1])),
        y2='y2:Q',
        color=alt.Color('residual:Q',
                        scale=alt.Scale(domain=[-max_abs_residual, max_abs_residual],
                                        range=['#2166ac', '#f7f7f7', '#b2182b']),
                        legend=alt.Legend(title='Pearson Residuals',
                                        orient='right',
                                        fillColor='rgba(255, 255, 255, 0.7)',
                                        offset=-130,
                                        padding=10))
    ).properties(
        title=f"Mosaic Plot with Pearson Residuals: {selected_feature.title()} Distribution",
        height=600,
    )
    
    # Create text labels for demographic values
    text_labels = alt.Chart(label_df).mark_text(
        align='left',
        dx=5,
        fontSize=12,
    ).encode(
        x=alt.X('x_mid:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('y_mid:Q', scale=alt.Scale(domain=[0, 1])),
        text='feature:N'
    )
    
    # Combine mosaic plot with text labels
    mosaic_plot = base_mosaic.mark_rect() + text_labels
    st.altair_chart(mosaic_plot)

    
render_navigation("features/index.py", "features/admission.py")
