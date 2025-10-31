from typing import TypedDict, Annotated , Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage # The fondational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data black to LLM after it calls a tool such as the content and and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# Annotated - provides additional context without affecting the type itself
# Sequence - To automatically handle the state updates for sequences such as by adding new messages to a chat history
email = Annotated[str, "This has to be a valid email format!"]
print(email.__metadata__)

