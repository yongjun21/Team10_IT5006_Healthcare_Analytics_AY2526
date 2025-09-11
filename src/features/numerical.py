import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.graph_objects as go

from helper import render_navigation, get_scatter_data

data = st.session_state.data
outcome_oh = st.session_state.outcome_oh

st.title("Feature Analysis - Numerical Features")


st.subheader("Distribution of Length of Stay")

# Length of Stay violin plot
fig, ax = plt.subplots(figsize=(12, 6))
jitter = np.random.uniform(-0.5, 0.5, len(data))
# Ensure proper ordering: <30, >30, NO
data_ordered = data.copy()
data_ordered['readmitted'] = pd.Categorical(data_ordered['readmitted'],
                                            categories=['<30', '>30', 'NO'],
                                            ordered=True)

st.markdown('##### Length of Stay Distribution by Readmission Status')
sns.violinplot(data=data_ordered, x='readmitted',
               y=data_ordered['time_in_hospital'] + jitter, ax=ax)
ax.set_xlabel('Readmission')
ax.set_ylabel('Length of Stay (days)')
st.pyplot(fig, use_container_width=False)
plt.close()


st.subheader("Relation between Number of Past Visits and Readmission Outcome")

# Number of Past Visits violin plots
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

visit_types = ['number_outpatient', 'number_inpatient', 'number_emergency']
titles = ['Outpatient Visits', 'Inpatient Visits', 'Emergency Visits']

# First melt the data
melted_data = data.melt(
    id_vars=['readmitted'],
    value_vars=['number_outpatient', 'number_inpatient', 'number_emergency'],
    var_name='visit_type',
    value_name='number_of_visits'
)

# Add random jitter to reduce overlap
melted_data['number_of_visits'] = melted_data['number_of_visits'] + \
    np.random.uniform(-0.5, 0.5, len(melted_data))

filtered_melted = melted_data[melted_data['number_of_visits'] > 0]
percentile_99 = filtered_melted.groupby(
    'visit_type')['number_of_visits'].transform(lambda x: x.quantile(0.99))
filtered_melted = filtered_melted[filtered_melted['number_of_visits']
                                  <= percentile_99]

st.markdown('##### Distribution of Number of Past Visits by Readmission Status')
for i, (visit_type, title) in enumerate(zip(visit_types, titles)):
    subset = filtered_melted[filtered_melted['visit_type'] == visit_type]

    sns.violinplot(data=subset,
                   x='readmitted',
                   y='number_of_visits',
                   ax=axes[i],
                   inner='box')
    axes[i].set_ylim(-0.5, 5.5)
    axes[i].set_title(title, fontsize=14)
    axes[i].set_xlabel('Readmission Status', fontsize=12)
    axes[i].set_ylabel('Number of Visits', fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
st.pyplot(fig)
plt.close()


st.subheader("Relation between Number of Past Visits and Readmission Outcome")

x, y = get_scatter_data(data, ['number_outpatient'], [
                        'number_inpatient', 'number_emergency'])
hue = data['readmitted'].map(
    {'<30': '#1f77b4', '>30': '#ff7f0e', 'NO': '#2ca02c'})

fig = go.Figure(
    data=[
        go.Scattergl(
            x=x[hue == '#1f77b4'], y=y[hue == '#1f77b4'],
            mode="markers",
            marker=dict(size=3, color='#1f77b4'),
            name="<30",
            showlegend=True,
        ),
        go.Scattergl(
            x=x[hue == '#ff7f0e'], y=y[hue == '#ff7f0e'],
            mode="markers",
            marker=dict(size=3, color='#ff7f0e'),
            name=">30",
            showlegend=True,
        ),
        go.Scattergl(
            x=x[hue == '#2ca02c'], y=y[hue == '#2ca02c'],
            mode="markers",
            marker=dict(size=3, color='#2ca02c'),
            name="NO",
            showlegend=True,
        )
    ],
    layout=go.Layout(
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(title="Number of Outpatient Visits", range=[0, 20]),
        yaxis=dict(title="Number of Inpatient + Emergency Visits",
                   range=[0, 30]),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="right",
            x=0.95
        )
    )
)

st.markdown(
    '##### Scatter Plot of Number of Outpatient Visits vs Number of Inpatient + Emergency Visits')
st.plotly_chart(fig, use_container_width=True)


st.subheader(
    "Relation between Number of Medications Prescribed and Length of Stay")

x, y = get_scatter_data(data, ['num_medications'], ['time_in_hospital'])
hue = data['readmitted'].map(
    {'<30': '#1f77b4', '>30': '#ff7f0e', 'NO': '#2ca02c'})

fig = go.Figure(
    data=[
        go.Scattergl(
            x=x[hue == '#2ca02c'], y=y[hue == '#2ca02c'],
            mode="markers",
            marker=dict(size=3, color='#2ca02c'),
            name="NO",
            showlegend=True,
        ),
        go.Scattergl(
            x=x[hue == '#ff7f0e'], y=y[hue == '#ff7f0e'],
            mode="markers",
            marker=dict(size=3, color='#ff7f0e'),
            name=">30",
            showlegend=True,
        ),
        go.Scattergl(
            x=x[hue == '#1f77b4'], y=y[hue == '#1f77b4'],
            mode="markers",
            marker=dict(size=3, color='#1f77b4'),
            name="<30",
            showlegend=True,
        )
    ],
    layout=go.Layout(
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(title="Number of Medications Prescribed"),
        yaxis=dict(title="Length of Stay (days)"),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="right",
            x=0.95
        )
    )
)

st.markdown(
    '##### Scatter Plot of Number of Outpatient Visits vs Number of Inpatient + Emergency Visits')
st.plotly_chart(fig, use_container_width=True)


st.subheader("Correlation Analysis")

numeric_cols = ['time_in_hospital',
                'number_outpatient', 'number_inpatient', 'number_emergency',
                'num_procedures', 'num_lab_procedures', 'num_medications', 'number_diagnoses']

corr_matrix = pd.concat([outcome_oh["readmitted_<30"],
                        data[numeric_cols]], axis=1).corr()

# Prepare data for Altair heatmap
corr_data = []
feature_order = list(corr_matrix.index)

for i, row in enumerate(corr_matrix.index):
    for j, col in enumerate(corr_matrix.columns):
        if i != j:  # Skip diagonal
            corr_data.append({
                'feature1': row,
                'feature2': col,
                'correlation': corr_matrix.iloc[i, j]
            })

corr_df = pd.DataFrame(corr_data)

heatmap = alt.Chart(corr_df).mark_rect().encode(
    x=alt.X('feature1:N', title='Features', sort=feature_order,
            axis=alt.Axis(labelAngle=-90, labelLimit=200)),
    y=alt.Y('feature2:N', title='Features', sort=feature_order,
            axis=alt.Axis(labelPadding=20, labelLimit=200)),
    color=alt.Color('correlation:Q',
                    scale=alt.Scale(domain=[-0.5, 0.5],
                                    range=['#2166ac', '#f7f7f7', '#b2182b']),
                    legend=alt.Legend(title='Correlation')),
    tooltip=[alt.Tooltip('feature1:N', title='Feature 1'),
             alt.Tooltip('feature2:N', title='Feature 2'),
             alt.Tooltip('correlation:Q', title='Correlation', format='.3f')]
).properties(
    width=600,
    height=600,
    title='Correlation Heatmap of Numerical Features'
)

# Add text annotations
text = alt.Chart(corr_df).mark_text(
    align='center',
    baseline='middle',
    fontSize=10
).encode(
    x=alt.X('feature1:N', sort=feature_order,
            axis=alt.Axis(labelAngle=-90, labelLimit=200)),
    y=alt.Y('feature2:N', sort=feature_order,
            axis=alt.Axis(labelLimit=200)),
    text=alt.Text('correlation:Q', format='.2f'),
    color=alt.condition(
        abs(alt.datum.correlation) > 0.25,
        alt.value('white'),
        alt.value('black')
    )
)

correlation_chart = (heatmap + text).resolve_scale(color='independent')

st.altair_chart(correlation_chart)

render_navigation("features/admission.py", "features/medications.py")
