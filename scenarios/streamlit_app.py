# streamlit_app.py
import streamlit as st
from app import answer_question  # import your function

st.title("Test Scenario Assistant")

question = st.text_area(
    "Ask about requirements or ask for test scenarios:",
    "Generate test scenarios for the login feature."
)

if st.button("Generate"):
    with st.spinner("Thinking..."):
        answer = answer_question(question)
    st.markdown(answer)
