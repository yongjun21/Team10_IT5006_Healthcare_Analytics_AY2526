import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from helper import render_navigation, get_outcome_by_feature

data = st.session_state.data

data["max_glu_serum"] = data["max_glu_serum"].cat.reorder_categories(['Norm', '>200', '>300'])
data["A1Cresult"] = data["A1Cresult"].cat.reorder_categories(['Norm', '>7', '>8'])

@st.cache_data
def get_chart_data(outcome_by_feature, selected_feature):
    feature_order = outcome_by_feature.index.tolist()

    chart_data = outcome_by_feature.reset_index().melt(
        id_vars=[selected_feature],
        var_name='readmitted',
        value_name='count'
    )
    
    # Clean up the outcome column names
    chart_data['readmitted'] = chart_data['readmitted'].str.replace('readmitted_', '')
    
    return chart_data, feature_order

def render_distribution_chart(data, selected_feature, chart_type, title):
    outcome_by_feature = get_outcome_by_feature(data, None, selected_feature)

    chart_data, feature_order = get_chart_data(outcome_by_feature, selected_feature)

    if chart_type == "Count":

        # Create grouped bar chart
        base = alt.Chart(chart_data).encode(
            x=alt.X(f'{selected_feature}:O',
                    title=title,
                    axis=alt.Axis(labelAngle=-45),
                    sort=feature_order),
            y=alt.Y('count:Q', title='Count'),
            xOffset='readmitted:N',
            color=alt.Color('readmitted:N',
                            scale=alt.Scale(domain=['<30', '>30', 'NO'],
                                            range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                            legend=alt.Legend(title='Readmitted'))
        ).properties(
            height=500
        )

        st.markdown(f'##### Readmitted Distribution by {title}')
        bars = base.mark_bar()
        st.altair_chart(bars)

    elif chart_type == "Proportion":

        # Create horizontal stacked bar chart
        base_pct = alt.Chart(chart_data).encode(
            x=alt.X('count:Q', title='Percentage (%)',
                    stack="normalize",
                    axis=alt.Axis(format='%'),
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(f'{selected_feature}:O',
                    title=title,
                    axis=alt.Axis(format='%'),
                    sort=feature_order),
            color=alt.Color('readmitted:N',
                            scale=alt.Scale(domain=['<30', '>30', 'NO'],
                                            range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                            legend=alt.Legend(title='Readmitted'))
        ).properties(
            height=250
        )

        st.markdown(f'##### Readmitted Distribution by Max Glucose in Serum')
        bars = base_pct.mark_bar()
        st.altair_chart(bars)

def render_chi_square_test(selected_feature):
    contingency_table = pd.crosstab(
        data[selected_feature], 
        data['readmitted']
    )

    chi2, p_value, dof, _ = chi2_contingency(contingency_table)

    chi_square_test_info = pd.DataFrame({
        "key": ["Chi-square statistic", "p-value", "Degrees of freedom"],
        "value": [f" {chi2:.4f}", f"{p_value:.4f}", dof]
    })
    st.dataframe(chi_square_test_info, hide_index=True, use_container_width=False)

    st.text("Contingency table:")
    # Rename columns to avoid HTML formatting issues
    display_table = contingency_table.copy()
    display_table.columns = [col.replace('<', '&lt; ').replace('>', '&gt; ') for col in display_table.columns]
    display_table.index = [index.replace('<', '&lt; ').replace('>', '&gt; ') for index in display_table.index]
    st.table(display_table.round().astype(int))

st.title("Feature Analysis - Lab Results")

st.subheader("Distribution of Readmitted by Max Glucose in Serum")

chart_type = st.selectbox("View As", ["Count", "Proportion"], key="max_glu_serum")

render_distribution_chart(data, "max_glu_serum", chart_type, "Max Glucose in Serum")

st.markdown(f"##### Chi-square test for Max Glucose in Serum vs Readmission Outcome")

render_chi_square_test("max_glu_serum")


st.subheader("Distribution of Readmitted by Max Glucose in Serum")

chart_type_2 = st.selectbox("View As", ["Count", "Proportion"], key="A1Cresult")

render_distribution_chart(data, "A1Cresult", chart_type_2, "A1C Result")

render_chi_square_test("A1Cresult")

st.markdown("**Observation** Chi Square Test shows both `max_glu_serum` and `A1Cresult` have statistically significant effect on readmission risk.")

render_navigation("features/diagnoses.py", "mixed.py")
