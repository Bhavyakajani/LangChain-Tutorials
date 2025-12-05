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

![img.png](chatbot/img.png)

### File Upload using LangChain

In this step, we introduce **document processing** using LangChain’s PDF loader.  
The goal is simple:

> **Upload a PDF → Extract its text → Generate test scenarios + edge cases using an LLM.**

This marks the transition from simple LLM chat to building an actual **QA assistant**.

---

## 📄 1. Load the PDF

We use `PyPDFLoader` from `langchain_community`:

```python
from langchain_community.document_loaders import PyPDFLoader

pages = PyPDFLoader("documents/user_story.pdf").load()
text = "\n".join([p.page_content for p in pages])
```

---

## ✂️ 2. Split the text into chunks

Large documents need to be split for effective processing:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)

chunks = splitter.split_text(text)
```

---

## 🤖 3. Generate Test Scenarios Using LLM + Prompt

We define a **FORMAT** (JSON schema) and a custom prompt instructing the model to generate:

- Positive test scenarios  
- Negative test scenarios  
- Edge case scenarios  

The prompt ensures **consistent structure** across outputs.

**Simplified example prompt:**

```python
prompt = PromptTemplate(
    input_variables=["FORMAT", "CONTEXT"],
    template="""
You are a senior QA engineer.

Using the CONTEXT from the PDF, generate:
- Test scenarios (positive + negative)
- Edge case scenarios

Follow the JSON FORMAT strictly.

FORMAT:
{FORMAT}

CONTEXT:
{CONTEXT}
"""
)
```

---

## 🔗 LLM Pipeline Using LCEL

LangChain Expression Language (LCEL) allows chaining components cleanly:

```python
chain = prompt | llm | JsonOutputParser()
response_json = chain.invoke({"FORMAT": FORMAT, "CONTEXT": CONTEXT})
```


Output
```commandline
{
  "test_scenarios": [
    {
      "id": "TS-1",
      "title": "Reindeer runs over water without sinking or slowing down",
      "type": "positive",
      "preconditions": [
        "Warm-climate region",
        "Foggy weather"
      ],
      "steps": [
        "Santa's reindeer is deployed in a warm-climate region",
        "A water body is detected ahead during foggy weather",
        "The system activates Water Run Mode",
        "The reindeer begins crossing the water"
      ],
      "expected_result": "The reindeer runs over the surface without sinking or slowing down"
    },
    {
      "id": "TS-2",
      "title": "Water Run Mode is activated correctly",
      "type": "positive",
      "preconditions": [
        "Warm-climate region",
        "Foggy weather"
      ],
      "steps": [
        "Santa's reindeer is deployed in a warm-climate region",
        "A water body is detected ahead during foggy weather"
      ],
      "expected_result": "The system activates Water Run Mode"
    },
    {
      "id": "TS-3",
      "title": "Route optimization works correctly",
      "type": "positive",
      "preconditions": [
        "Water body lies between Santa and the destination"
      ],
      "steps": [
        "A water body lies between Santa and the destination",
        "Water Run Mode is available"
      ],
      "expected_result": "The route is optimized to cross the water instead of going around it"
    },
    {
      "id": "TS-4",
      "title": "Reindeer gets stuck in water when Water Run Mode is not activated correctly",
      "type": "negative",
      "preconditions": [
        "Warm-climate region",
        "Foggy weather"
      ],
      "steps": [
        "Santa's reindeer is deployed in a warm-climate region",
        "A water body is detected ahead during foggy weather"
      ],
      "expected_result": "The system does not activate Water Run Mode"
    },
    {
      "id": "TS-5",
      "title": "Route optimization fails when Water Run Mode is not available",
      "type": "negative",
      "preconditions": [
        "Water body lies between Santa and the destination"
      ],
      "steps": [
        "A water body lies between Santa and the destination"
      ],
      "expected_result": "The route is not optimized to cross the water instead of going around it"
    }
  ],
  "edge_case_scenarios": [
    {
      "id": "EC-1",
      "title": "Reindeer runs over ice when Water Run Mode is not activated correctly",
      "preconditions": [
        "Cold-climate region"
      ],
      "steps": [
        "Santa's reindeer is deployed in a cold-climate region"
      ],
      "expected_result": "The system does not activate Water Run Mode"
    },
    {
      "id": "EC-2",
      "title": "Reindeer cannot fit through water when route optimization fails",
      "preconditions": [
        "Water body lies between Santa and the destination"
      ],
      "steps": [
        "A water body lies between Santa and the destination",
        "The system fails to optimize the route"
      ],
      "expected_result": "The reindeer cannot fit through the water"
    }
  ]
}
```