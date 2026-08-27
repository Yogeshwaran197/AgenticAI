import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Sequence, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
)

class ChatState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


def assistant(state : ChatState) -> ChatState:

    messages = state['messages']

    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a highly capable personal AI assistant. 
    Your owner is Yogi. 

    Rules and Behavior:
    - Prioritize Yogi's requests above all else.
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


graph = StateGraph(ChatState)

graph.add_node("LLM", assistant)

graph.add_edge(START, "LLM")
graph.add_edge("LLM", END)

checkpointer = InMemorySaver()

ChatBot = graph.compile(checkpointer=checkpointer)

config = {"configurable" : {
    "thread_id" : "yogi"
}}

while True:

    user_input = input("\nHuman : ")

    if user_input == "exit":
        break

    response = ChatBot.invoke({
        "messages": user_input
    }, config=config)

    print(f"\nAI : {response['messages'][-1].content}")




