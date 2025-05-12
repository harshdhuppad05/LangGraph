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
    
    pass


def logical_agent(state:State):
    pass

def router(state: State):
    message_type = state.get('message_type', 'logical')
    if message_type == "emotional":
        return {"next":"therapist"}
    return {"next":"logical"}


graph_builder = StateGraph(State)


