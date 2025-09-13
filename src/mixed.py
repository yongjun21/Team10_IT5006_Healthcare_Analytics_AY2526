import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
        else:
            exog = sm.add_constant(data[predictor])

    model = MixedLM(
        endog=data[outcome],
        exog=exog,
        groups=data['patient_nbr'],
        exog_re=np.ones((len(exog), 1))
    )

    result = model.fit()
    random_effects = result.cov_re.iloc[0,0]  # Intercept variance
    residual = result.scale
    return random_effects / (random_effects + residual)

@st.cache_data
def compute_full_model_icc(data, outcome, predictors):
    exog_components = []
    
    for predictor in predictors:
        if isinstance(data[predictor].dtype, pd.CategoricalDtype) or data[predictor].dtype == object:
            # Handle categorical variables
            if data[predictor].isna().any():
                if isinstance(data[predictor].dtype, pd.CategoricalDtype):
                    if 'Unknown' not in data[predictor].cat.categories:
                        data[predictor] = data[predictor].cat.add_categories('Unknown')
                    clean_data = data[predictor].fillna('Unknown')
                else:
                    clean_data = data[predictor].fillna('Unknown')
            else:
                clean_data = data[predictor]
            
            # Create dummy variables
            dummies = pd.get_dummies(clean_data, drop_first=True, dtype=int)
            exog_components.append(dummies)
            
        else:
            # Handle continuous variables
            continuous_data = data[predictor].fillna(data[predictor].mean())  # Fill with mean
            exog_components.append(continuous_data)
    
    # Concatenate all components
    if exog_components:
        exog = pd.concat(exog_components, axis=1)
    else:
        # If no predictors, use just intercept
        exog = pd.DataFrame({'const': np.ones(len(data))}, index=data.index)
    
    # Add constant for the intercept (if not already present from continuous variables)
    exog = sm.add_constant(exog)
    
    model = MixedLM(
        endog=data[outcome],
        exog=exog,
        groups=data['patient_nbr'],
        exog_re=np.ones((len(exog), 1))
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

st.text("Mixed Model Analysis allows us to study how much unaccounted patient level differences influence outcomes thus guiding modelling approach.")

# progress_bar = st.progress(0.0)

# icc_results = {}
# icc_results['NULL'] = compute_icc(data, 'time_in_hospital', None)
# progress_bar.progress(1.0 / (len(test_predictors) + 2))
# for i, predictor in enumerate(test_predictors):
#     icc_results[predictor] = compute_icc(data, 'time_in_hospital', predictor)
#     progress_bar.progress(((i + 2.0) / (len(test_predictors) + 2)))
# icc_results['FULL'] = compute_full_model_icc(data, 'time_in_hospital', test_predictors)
# progress_bar.progress(1.0)
# progress_bar.empty()

icc_results_los = pickle.load(open('src/assets/icc_results_los.pkl', 'rb'))
chart = render_icc_chart(icc_results_los)

st.markdown('##### ICC for Length of Stay by Predictor')
st.altair_chart(chart, use_container_width=True)

st.markdown("**Observation** For predicting length of stay, the null model's intraclass correlation coefficient is rather high at 0.19. This value did not go down significantly when we add fixed effects to the model. In fact ICC goes up when we add in the number of procedures or number of medications as fixed effects. This shows factors like the number of medications only help explain within group variation but have smaller or no effect on between group variation. Thus patient level differences persist and if not accounted for in modelling will lead to inflation of the other model features’ fixed effects.")

# progress_bar = st.progress(0.0)

# data_with_oh = pd.concat([data, outcome_oh], axis=1)

# icc_results = {}
# icc_results['NULL'] = compute_icc(data_with_oh, 'readmitted_<30', None)
# progress_bar.progress(1.0 / (len(test_predictors) + 1))
# for i, predictor in enumerate(test_predictors):
#     icc_results[predictor] = compute_icc(data_with_oh, 'readmitted_<30', predictor)
#     progress_bar.progress(((i + 2.0) / (len(test_predictors) + 1)))
# icc_results['FULL'] = compute_full_model_icc(data_with_oh, 'readmitted_<30', test_predictors)
# progress_bar.progress(1.0)
# progress_bar.empty()

icc_results_readmitted = pickle.load(open('src/assets/icc_results_readmitted.pkl', 'rb'))
chart = render_icc_chart(icc_results_readmitted)

st.markdown('##### ICC for Readmitted in less than 30 days by Predictor')
st.altair_chart(chart, use_container_width=True)

st.markdown("**Observation** The story is different for prediction of readmission risk. The null model's ICC is relatively low at 0.07. After including the number of past inpatient visits as fixed effect, ICC shrinks to 0.016. Thus we are able to account for most of the between group variation so patient level differences can be ignored without much impact to model generalizability, robustness and interpretability.")


render_navigation("features/lab.py", None)
