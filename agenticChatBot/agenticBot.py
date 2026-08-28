from langgraph.checkpoint import serde
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Sequence, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.postgres import  PostgresSaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from psycopg import Connection
from psycopg.rows import dict_row


load_dotenv()

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
)

class ChatState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


def assistant(state : ChatState) -> ChatState:

    messages = state['messages']

    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a highly capable personal AI assistant. . 

    Rules and Behavior:
    - Maintain a helpful, smart, and efficient tone.
    - Adapt your style to match Yogi's mood and needs.
    - Keep answers concise, actionable, and straight to the point.
    - If anyone else interacts with you, remain polite but protect Yogi's privacy.
    """),
    MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | llm

    response = chain.invoke({"messages" : messages})

    return {"messages" : [response]}


serde = EncryptedSerializer.from_pycryptodome_aes()

conn = "postgresql://yogi:3122005@localhost:5432/chatbot"

pg_conn = Connection.connect(conn, autocommit=True, row_factory=dict_row)
checkpointer = PostgresSaver(pg_conn, serde=serde)
checkpointer.setup()


graph = StateGraph(ChatState)

graph.add_node("LLM", assistant)

graph.add_edge(START, "LLM")
graph.add_edge("LLM", END)

ChatBot = graph.compile(checkpointer=checkpointer)

config = {"configurable" : {
    "thread_id" : "yogi"
}}

while True:

    user_input = input("\nHuman : ")

    if user_input == "exit":
        break

    for message_chunk, metadata in ChatBot.stream({
        "messages": [HumanMessage(content=user_input)]
    }, config=config, stream_mode="messages"):
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)
    print()