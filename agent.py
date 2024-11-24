from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3", request_timeout=360.0)
prompt = ChatPromptTemplate.from_template("You`re a helpful agent, just talk with the user about what he wants: {topic}")

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    chatbot_runnable = prompt | llm

    return {"messages": [chatbot_runnable.invoke(state["messages"])]}

graph_builder = StateGraph(State)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()
