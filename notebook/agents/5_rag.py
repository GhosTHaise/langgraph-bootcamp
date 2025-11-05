from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph.message import add_messages

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0)

# Our Embedding Model - has to also be compatible with the LLM -> gemini-embedding-001
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

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

# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

pages_split = text_splitter.split_documents(pages)

persist_directory = "notebook/agents/db"
collection_name = "stock_market"

# create the folder if not exist
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print(f"Chroma DB created at {persist_directory}")
    
except Exception as e:
    print(f"Error creating Chroma DB: {e}")
    raise e


# create a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

@tool
def retriever_tool(query: str) -> str:
    """
        This tool searches and returns the information from {pdf_path} document.
    """
    
    docs = retriever.invoke(query)
    
    if not docs:
        return f"I found not relevant information in the {pdf_path} document."
    
    results = []
    for i , doc in enumerate(docs):
        results.append(f"Document {i+1}: {doc.page_content}")
        
    return "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages = Annotated[Sequence[BaseMessage], add_messages]
    
def should_continue(state: AgentState):
    """Check if the last message contains tools calls"""
    
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about graphic cards based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the graphic cards data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

def call_llm(state: AgentState) -> AgentState:
    print(state)
    messages = list(state.get("messages", []))
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {"messages": [AIMessage(content=message.content)]}

# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    print(state)
    last_message = state.get("messages", [])[-1]
    
    # Vérifie si l'objet a bien des tool_calls
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        print("No tool calls found. Skipping retriever_agent.")
        return {"messages": []}
    
    tool_calls = last_message.tool_calls
    results = []
    
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.set_entry_point("llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True : "retriever_agent", False : END}
)
graph.add_edge("retriever_agent", "llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages" : messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()