# app.py
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

# 1. LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# 2. Load PDF requirements
loader = PyPDFLoader("requirements.pdf")
docs = loader.load()

# 3. Split + store
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(chunks, embedding=embeddings)
retriever = vectordb.as_retriever()

# 4. Prompt
prompt = ChatPromptTemplate.from_template("""
You are a senior QA engineer helping to design tests.

Context from the requirements:
{context}

User request:
{question}

Respond with:
- Well-structured test scenarios
- Clear titles
- Positive, negative and edge cases where relevant.
""")

chain = prompt | llm | StrOutputParser()

def answer_question(question: str) -> str:
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(d.page_content for d in docs)
    return chain.invoke({"context": context, "question": question})

if __name__ == "__main__":
    # quick CLI test
    q = "Generate test scenarios for the login feature."
    print(answer_question(q))
