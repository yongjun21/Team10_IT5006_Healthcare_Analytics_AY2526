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

render_navigation("data.py", "features/index.py")
