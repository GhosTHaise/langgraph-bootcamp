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

""" | Élément                      | Rôle principal                | Exemple rapide                     |
| ---------------------------- | ----------------------------- | ---------------------------------- |
| `TypedDict`                  | Dictionnaire typé             | `class User(TypedDict): name: str` |
| `Annotated`                  | Métadonnées sur un type       | `Annotated[str, "user name"]`      |
| `Sequence`                   | Liste ou tuple typé           | `Sequence[int]`                    |
| `load_dotenv`                | Charger `.env`                | `load_dotenv()`                    |
| `HumanMessage`               | Message humain                | `HumanMessage("Hello")`            |
| `SystemMessage`              | Règles pour le LLM            | `SystemMessage("You are helpful")` |
| `ToolMessage`                | Résultat d’un outil           | `ToolMessage("Done", "tool_1")`    |
| `StateGraph`, `START`, `END` | Construire un graphe d’états  | `graph.add_edge(START, END)`       |
| `ChatGroq`                   | LLM basé sur Groq             | `ChatGroq(model="mixtral-8x7b")`   |
| `@tool`                      | Déclare un outil              | `@tool def add(a,b): return a+b`   |
| `add_messages`               | Ajoute des messages à un état | `add_messages(history, [msg])`     |
| `ToolNode`                   | Nœud prêt pour les outils     | `ToolNode(tools=[my_tool])`        |
 """
 
def decorateur(func):
    def wrapper():
        print("Avant la fonction")
        func()  # exécute la fonction originale
        print("Après la fonction")
    return wrapper

@decorateur
def dire_bonjour():
    print("Bonjour !")

dire_bonjour()


class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]
    
@tool
def add(a : int , b : int):
    """ This is an addition function that adds two numbers together """
    
    return a + b

@tool
def multiplication(a : int , b : int):
    """ This is an multiplication function that multiplies two numbers together """
    
    return a + b

@tool
def subtraction(a : int , b : int):
    """ subtraction function that subtracts two numbers """
    
    return a + b

tools = [add,multiplication,subtraction]

model = ChatGroq(model="llama-3.1-8b-instant").bind_tools(tools)


def model_call(state : AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])
    
    return {"messages" : [response]}  

def should_continue(state : AgentState):
    messages = state["messages"]
    
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue" : "tools",
        "end" : END
    }
)


graph.add_edge("tools", "our_agent")

app= graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        
        else:
            message.pretty_print()
            

inputs = {"messages" : [("user", "Add 41 + 12 , 4 - 5 , 3 * 534 , return the result")]}
#inputs2 = {"messages" : [("user","What is my last query ?")]}

print_stream(app.stream(inputs, stream_mode="values"))
#print_stream(app.stream(inputs2, stream_mode="values"))