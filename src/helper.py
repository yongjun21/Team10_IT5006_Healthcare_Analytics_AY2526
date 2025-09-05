import streamlit as st

def render_navigation(previous_page, next_page):
    st.markdown("---")

    if previous_page is not None and next_page is not None:
        container = st.container(horizontal=True, horizontal_alignment="distribute")
        container.page_link(previous_page, label="Previous", icon=":material/chevron_left:")
        container.page_link(next_page, label="Next", icon=":material/chevron_right:")
        return
    
    if next_page is not None:
        container = st.container(horizontal=True, horizontal_alignment="right")
        container.page_link(next_page, label="Next", icon=":material/chevron_right:")
        return

    if previous_page is not None:
        st.page_link(previous_page, label="Previous", icon=":material/chevron_left:")
        return
