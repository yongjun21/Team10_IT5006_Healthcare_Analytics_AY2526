import streamlit as st
import altair as alt
import pandas as pd

from helper import render_navigation, get_by_patient

st.title("By Patient")

data = st.session_state.data
outcome_oh = st.session_state.outcome_oh

if "by_patient" not in st.session_state:
    st.session_state.by_patient = get_by_patient(data, outcome_oh)
by_patient = st.session_state.by_patient

max_patient_encounters = by_patient["encounters"].max()

st.subheader("Distribution of Patient Encounters")

base = alt.Chart(by_patient.reset_index()).encode(
    x=alt.X('encounters', bin=alt.Bin(maxbins=max_patient_encounters), title='Number of Encounters'),
    y=alt.Y('count()', scale=alt.Scale(type='log'), title='Count of Patients')
).properties(
    height=600
)

bars = base.mark_bar()
text = base.mark_text(
    align='center',
    baseline='bottom',
    dy=-5
).encode(
    text=alt.Text('count():Q')
)

st.altair_chart(bars + text)

st.text("Distribution Properties")
st.dataframe(by_patient["encounters"].describe(), use_container_width=False)

st.subheader("Patient With The Most Encounters")

rep_max_patient = by_patient[by_patient["encounters"] == max_patient_encounters].index[0]
rep_max_patient_data = data[data["patient_nbr"] == rep_max_patient]

st.dataframe(rep_max_patient_data)

st.markdown("**Observation** Even though the data unit in this study is the hospitalization encounter, it is useful to also analyze from the perspective of the patient. There were significant repeated appearances of the same patient within the dataset with the largest count at 40. The ~100K hospitalization encounters are distributed between ~70K patients. With the distribution of repeated patient encounters exhibiting a power law relationship. Because certain features like demographic features are duplicated within encounters for a single patient while others like number of past visits, prescribed medications etc are highly correlated, the implication for modelling is:\n1. Train/Test split should avoid information leakage via patient (i.e. the same patient should not be in both train and test set)\n2. Mixed Model approach might be useful here to separate out the effect of patient level differences from the true effect of individual factors.")

render_navigation("data.py", "features/index.py")
