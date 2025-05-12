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


class Message_Classifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(..., description="classify if the message requires emotional(therapist) or logical response",alias="message_type")


class State(TypedDict):
    messages: Annotated[str, add_messages]
    message_type: str | None
    next : str|None

def classify_message(state:State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(Message_Classifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content":"""
                classify the user message as either:
                - 'emothional' : if it asks for emotional support, therapy, deal with feelings, or personal problems
                - 'logical': if it asks for facts, information, logical analysis or practical solutions
            """
        },
        {
            "role":"user", "content": last_message.content
        }
    ])
    return {"message_type":result.message_type}
    

def therapist_agent(state:State):
    last_message = state["messages"][-1]
    messages = [{"role": "system",
         "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def logical_agent(state:State):
    last_message = state["messages"][-1]
    messages = [
        {"role": "system",
         "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def router(state: State):
    message_type = state.get('message_type', 'logical')
    if message_type == "emotional":
        return {"next":"therapist"}
    return {"next":"logical"}


graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("logical",logical_agent)
graph_builder.add_node("therapist", therapist_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"therapist": "therapist", "logical": "logical"}
    )

graph_builder.add_edge("logical", END)
graph_builder.add_edge("therapist", END)


graph_builder.compile()

