from langchain_ollama import ChatOllama
import streamlit as st

llm = ChatOllama(
    model="llama3.2"
)

st.title("Flying Llama")
question = st.text_input("Enter a question:")

if question and st.button("Ask"):
    with st.spinner("Thinking..."):
        try:
            response_ollama = llm.invoke(question)
            st.write(response_ollama.content)
        except Exception as e:
            st.error(f"Error: {e}")

