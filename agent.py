from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import PromptTemplate

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_openai import ChatOpenAI

def get_transcript(url: str) -> str:
    """Get yourube video transcript"""

    if url.find("youtube"):
        url = url.replace("https://www.youtube.com/watch?v=", "")

    transcript = YouTubeTranscriptApi.get_transcript(url)
    return transcript

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = PromptTemplate(
    template="""Show the transcript bellow 
            '{transcript}'""",
    input_variables=["transcript"],
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    transcript = get_transcript(state["messages"][-1].content)

    chatbot_runnable = prompt | llm
    return {"messages": [chatbot_runnable.invoke(transcript)]}

graph_builder = StateGraph(State)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()
