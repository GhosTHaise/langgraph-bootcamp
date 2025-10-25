from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]
    
llm = ChatGroq(model="llama-3.1-8b-instant")