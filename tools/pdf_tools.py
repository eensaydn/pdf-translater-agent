from langchain_core.tools import tool
from PyPDF2 import PdfReader
from fpdf import FPDF
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama  # Correct import for ChatOllama

@tool
def read_pdf(file_path: str) -> str:
    """
    Read the content of a PDF file and return the extracted text.
    """
    reader = PdfReader(file_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

@tool
def translate_to_turkish(text: str) -> str:
    """
    Translate the given English text into Turkish using local LLM.
    """
    llm = ChatOllama(
        model="llama2",
        base_url="http://localhost:11434"  # Local LLM servis adresini buraya yaz
    )
    message = HumanMessage(content=f"Translate the following English text to Turkish:\n\n{text}")
    result = llm.invoke([message]).content
    if isinstance(result, str):
        return result
    elif isinstance(result, list):
        return "\n".join(str(item) for item in result)
    else:
        return str(result)

@tool
def write_pdf(text: str, output_path: str = "translated.pdf") -> str:
    """
    Write the given text into a new PDF and return the path.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(output_path)
    return output_path

@tool
def summarize_text(text: str) -> str:
    """
    Generate a concise summary of the given text using local LLM.
    """
    llm = ChatGroq(
        model="llama2",
        base_url="http://localhost:your_groq_port",  # Groq local servis adresi
        temperature=0.3
    )
    prompt = f"Summarize the following text in a concise and clear way:\n\n{text}"
    result = llm.invoke([HumanMessage(content=prompt)]).content
    if isinstance(result, str):
        return result
    elif isinstance(result, list):
        return "\n".join(str(item) for item in result)
    else:
        return str(result)
