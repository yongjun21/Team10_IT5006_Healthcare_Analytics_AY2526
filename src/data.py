import streamlit as st
import altair as alt
import pandas as pd

st.title("Data Characteristics")

# drop the last 3 columns which are the outcome one-hot encoded
data = st.session_state.data.iloc[:,:-3]

st.subheader("Sample")
st.dataframe(data.head(10))

st.subheader("Distribution of Outcome")

data = st.session_state.data

data["readmitted"] = data["readmitted"].astype('category')
outcome_value_counts = data["readmitted"].value_counts()

base = alt.Chart(outcome_value_counts.reset_index()).encode(
    y="readmitted",
    x="count",
    color="readmitted"
).properties(
    height=200
)

bars = base.mark_bar()
text = base.mark_text(
    align='center',
    baseline='middle',
    dx=25
).encode(
    text=alt.Text('count:Q')
)

st.altair_chart(bars + text)

st.subheader("Completeness")
st.text("Percentage of missing values")

missing_count = data.isnull().sum()
missing_pct = (missing_count / len(data)) * 100
missing_stats = pd.DataFrame({
    'missing_count': missing_count,
    'missing_pct': missing_pct
})

base = alt.Chart(missing_stats[missing_stats['missing_pct'] > 0].reset_index()).encode(
    x=alt.X('index', title='Feature', axis=alt.Axis(labelAngle=-60, labelLimit=0), sort='-y'),
    y=alt.Y('missing_pct', title='Percentage of Missing Values'),
)

bars = base.mark_bar()
text = base.mark_text(
    align='center',
    baseline='bottom',
    dy=-5
).encode(
    text=alt.Text('missing_pct:Q', format='.1f')
).properties(
    height=600
)

st.altair_chart(bars + text)
