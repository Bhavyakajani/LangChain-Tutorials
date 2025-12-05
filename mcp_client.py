import asyncio
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient(
    {
        "tools": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http"
        }
    }
)

tools = asyncio.run(client.get_tools())

llm = ChatOllama(
    model="llama3.2:latest",
    base_url="http://localhost:11434"
    # bypass_streaming=True, # <-- avoids the streaming code path that raised the error
)
agent = create_agent(llm, tools)

st.title("AI Agent (MCP Version)")
task = st.text_input("Assign me a task")

if task:
    response = asyncio.run(agent.ainvoke({"message": task}))
    final_output = response["message"][-1].content
    st.write(final_output)