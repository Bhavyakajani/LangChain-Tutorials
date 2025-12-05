## Basics

### Lets start by connecting to an LLM

```commandline
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2"
)
response = llm.invoke("How do you explain to a kid that Santa isn't real?")
print(response.content)
```

### Now, time to make it interesting
We will now use ```streamlit``` library to create a frontend for the llm by adding the following snippet.
```commandline
st.title("Flying Llama")
question = st.text_input("Enter a question:")

if question and st.button("Ask"):
    try:
        response_ollama = llm.invoke(question)
        st.write(response_ollama.content)
    except Exception as e:
        st.error(f"Error: {e}")
```
And then we run the following command from terminal:

#### ```streamlit run filename.py```

![img.png](img.png)


### File Upload by using LangChain
