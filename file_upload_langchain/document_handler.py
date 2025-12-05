from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import sys
sys.path.append('../..')
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

UPLOAD_DIR = Path("uploaded_document")
UPLOAD_DIR.mkdir(exist_ok=True)

def get_chunks(text):
    """Split the extracted text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 200,
        length_function = len,
        separators = ["\n\n", "\n", " "],
    )
    chunks_without_split = text # to compare with the split chunks
    chunks = text_splitter.split_text(text)
    return chunks

def normalize_text(raw_output) -> str:
    if isinstance(raw_output, list):
        if all(hasattr(p, 'page_content') for p in raw_output):
            return "\n".join([p.page_content for p in raw_output])
        return "\n".join(raw_output)
    return str(raw_output)

def load_file(path: str):
    loader = PyPDFLoader(path)
    pages = loader.load()
    return pages

def get_llm():
    return ChatOllama(model="llama3.2")
def get_context(file_path: str) -> str:
    pages = load_file(file_path)
    context_text = normalize_text(pages)
    return context_text

def get_format() -> str:
    return """
{
  "test_scenarios": [
    {
      "id": "TS-1",
      "title": "Short title of the test scenario",
      "type": "positive | negative",
      "preconditions": ["list", "of", "preconditions"],
      "steps": ["step 1", "step 2", "step 3"],
      "expected_result": "Expected outcome of the scenario"
    }
  ],
  "edge_case_scenarios": [
    {
      "id": "EC-1",
      "title": "Short title of edge case scenario",
      "preconditions": ["list", "of", "preconditions"],
      "steps": ["step 1", "step 2"],
      "expected_result": "Expected outcome of the edge case"
    }
  ]
}
""".strip()

def get_prompt() -> str:
    return """
You are a senior QA engineer.

You will be given:
- FORMAT: a JSON schema describing how the output should look.
- CONTEXT: requirements text extracted from a PDF (user story, description, AC, etc.).

Using only the information from CONTEXT:
1. Identify the main behaviors and flows.
2. Generate a set of normal test scenarios (both positive and negative).
3. Generate a set of edge case scenarios (boundary, unusual inputs, system limits, etc.).
4. Follow the FORMAT exactly when structuring the JSON.

IMPORTANT:
- Use clear, concise titles.
- Steps must be action-based and sequential.
- Expected results must be explicit and testable.
- Do NOT invent fields that are not in the FORMAT.
- Return ONLY valid JSON, no extra commentary.

FORMAT:
{FORMAT}

CONTEXT:
{CONTEXT}
""".strip()


