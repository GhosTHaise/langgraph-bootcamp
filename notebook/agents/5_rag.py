from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0)

# Our Embedding Model - has to also be compatible with the LLM -> gemini-embedding-001
embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-ada-002")

pdf_path = "notebook/agents/pdf/GeForce-RTX-5070-12G-GAMING-TRIO-OC.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF Loaded with {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise e