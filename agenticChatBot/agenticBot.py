import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Sequence, Annotated
from langgraph.graph.message import add_messages



load_dotenv()

llm = ChatGroq(
    model = "qwen/qwen3.6-27b"
)

class ChatState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]