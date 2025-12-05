from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2"
)
response = llm.invoke("How do you explain to a kid that Santa isn't real?")
print(response.content)

