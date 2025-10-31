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