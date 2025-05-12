from dotenv import load_dotenv
import os
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END # type: ignore
from langgraph.graph.message import add_messages # type: ignore
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


load_dotenv()

llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

class State(TypedDict):
    messages: Annotated[str, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages" : [llm.invoke(state['messages'])]}

graph_builder.add_node("Chatbot",chatbot)

graph_builder.add_edge(START, "Chatbot")
graph_builder.add_edge("Chatbot", END)

graph = graph_builder.compile()

user_input = input("Enter your message: ")
state = graph.invoke({"messages":[{"role": "user", "content": user_input}]})

print(state['messages'][-1].content)