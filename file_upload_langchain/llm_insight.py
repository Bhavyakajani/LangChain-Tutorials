import json

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from document_handler import *



def extract_with_llm(file_path: str) -> dict:
    """
    Extract test scenarios and edge case scenarios from a PDF using an LLM.
    """
    FORMAT = get_format()
    CONTEXT = get_context(file_path)

    input_variables = ["FORMAT", "CONTEXT"]
    input_variables_dict = {
        "FORMAT": FORMAT,
        "CONTEXT": CONTEXT,
    }

    prompt = PromptTemplate(
        input_variables=input_variables,
        template=get_prompt(),
    )

    llm = get_llm()
    parser = JsonOutputParser()

    chain = prompt | llm | parser

    response_json = chain.invoke(input_variables_dict)
    return response_json

if __name__ == "__main__":
    pdf_path = "filepath.pdf"

    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    result = extract_with_llm(pdf_path)

    # Pretty-print the JSON so you can inspect / copy it
    print(json.dumps(result, indent=2, ensure_ascii=False))