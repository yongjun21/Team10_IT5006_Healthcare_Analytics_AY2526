import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

from helper import render_navigation

data = st.session_state.data
outcome_oh = st.session_state.outcome_oh

test_predictors = ['number_inpatient', 'num_procedures', 'num_lab_procedures',
                   'num_medications', 'number_diagnoses', 'race', 'gender', 'age']

@st.cache_data
def compute_icc(data, outcome, predictor):
    exog = np.ones(len(data))
    exog_re = None
    if predictor is not  None:
        if isinstance(data[predictor].dtype, pd.CategoricalDtype) or data[predictor].dtype == object:
            if data[predictor].isna().any():
                if isinstance(data[predictor].dtype, pd.CategoricalDtype):
                    if 'Unknown' not in data[predictor].cat.categories:
                        data[predictor] = data[predictor].cat.add_categories('Unknown')
                    clean_data = data[predictor].fillna('Unknown')
                else:
                    clean_data = data[predictor].fillna('Unknown')
            else:
                clean_data = data[predictor]
            exog = pd.get_dummies(clean_data, drop_first=False, dtype=int)
            exog_re = np.ones((len(exog), 1))
        else:
            exog = sm.add_constant(data[predictor])

    model = MixedLM(
        endog=data[outcome],
        exog=exog,
        groups=data['patient_nbr'],
        exog_re=exog_re
    )

    result = model.fit()
    random_effects = result.cov_re.iloc[0,0]  # Intercept variance
    residual = result.scale
    return random_effects / (random_effects + residual)

def render_icc_chart(icc_results):
    icc_df = pd.DataFrame({
        'predictor': list(icc_results.keys()),
        'icc': list(icc_results.values())
    })

    chart = alt.Chart(icc_df).mark_bar().encode(
        x=alt.X('predictor:N', 
                title='Predictor',
                scale=alt.Scale(domain=icc_df['predictor']),
                axis=alt.Axis(labelAngle=-90)),
        y=alt.Y('icc:Q',
                title='Intraclass Correlation Coefficient (ICC)',
                scale=alt.Scale(domain=[0, max(icc_results.values()) * 1.1])),
        tooltip=['predictor', alt.Tooltip('icc:Q', format='.3f')]
    ).properties(
        width=600,
        height=400
    )

    # Add text labels on top of bars
    text = alt.Chart(icc_df).mark_text(
        align='center',
        baseline='bottom',
        dy=-5  # Shift text up slightly from bar top
    ).encode(
        x='predictor:N',
        y='icc:Q',
        text=alt.Text('icc:Q', format='.3f')
    )

    return (chart + text).resolve_scale(color='independent')

st.title("Mixed Model Analysis")

progress_bar = st.progress(0.0)

icc_results = {}
icc_results['NULL'] = compute_icc(data, 'time_in_hospital', None)
progress_bar.progress(1.0 / (len(test_predictors) + 1))
for i, predictor in enumerate(test_predictors):
    icc_results[predictor] = compute_icc(data, 'time_in_hospital', predictor)
    progress_bar.progress(((i + 2.0) / (len(test_predictors) + 1)))
progress_bar.empty()

chart = render_icc_chart(icc_results)

st.markdown('##### ICC for Length of Stay by Predictor')
st.altair_chart(chart, use_container_width=True)

progress_bar = st.progress(0.0)

data_with_oh = pd.concat([data, outcome_oh], axis=1)

icc_results = {}
icc_results['NULL'] = compute_icc(data_with_oh, 'readmitted_<30', None)
progress_bar.progress(1.0 / (len(test_predictors) + 1))
for i, predictor in enumerate(test_predictors):
    icc_results[predictor] = compute_icc(data_with_oh, 'readmitted_<30', predictor)
    progress_bar.progress(((i + 2.0) / (len(test_predictors) + 1)))
progress_bar.empty()

chart = render_icc_chart(icc_results)

st.markdown('##### ICC for Readmitted in less than 30 days by Predictor')
st.altair_chart(chart, use_container_width=True)


render_navigation("features/quantitative.py", "features/medications.py")
