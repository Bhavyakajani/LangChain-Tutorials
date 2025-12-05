from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2:latest",          # use the exact name from `ollama list`
    base_url="http://localhost:11434",
    bypass_streaming=True,            # <- critical: avoid streaming path
)

resp = llm.invoke("Say only the word: hello")
print("Got from Ollama:", resp.content)
