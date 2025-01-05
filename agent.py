import re

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import PromptTemplate

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_openai import ChatOpenAI

def get_transcript(url: str) -> str:
    """Get yourube video transcript"""

    video_id = re.search("v=(\w+)&", url)

    transcript = YouTubeTranscriptApi.get_transcript(video_id.group(1), ["en", "fr"])
    return transcript

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = PromptTemplate(
    template="""Below is the transcription from a youtube video. 
            Your job is to
            extract a paragraph (seguence of sentences) from this transcript that is self-contained and explains a specific subject. 
            The paragraph should be roughly 500 words long. 
            Do not change the text format. 
            Do not explain to me why this paragraph was chosen. 
            Think slowly about wether the paragraph is self-contained or not and if it is 500 words long.
            If it doesn't fit these requirements, you can choose another paragraph that fits these requeriments better. 
            Return the paragraph in the following format:
            * start time: START_TIME_IN_MINUTES
            * end time: END_TIME_IN_MINUTES
            * paragraph: TEXT
            where START_TIME_IN_MINUTES is the first time from the first sentence in which the field 'start' is converted to HH:MM:SS, 
            END_TIME_IN_MINUTES is the last time from the last sentence in which the field 'start' is converted to HH:MM:SS
            TEXT is simply the text related to this paragraph without any aditions.

            Transcript: '{transcript}'""",
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
