from openai.types.responses import response
import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from  langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END


load_dotenv()

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
)

class AgentState(TypedDict):
    topic : str
    joke : str
    explanation: str


def llm_node(state:AgentState) -> AgentState:

    topic = state['topic']

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're Assistant, your job is to tell joke on topic"),
        ("user",  "heres a topic {topic} give joke based on  topic" ) 
    ])

    chain =  prompt | llm | StrOutputParser()

    response = chain.invoke({
        "topic" : topic
    })

    return {"joke" : response}


def explanation(state:AgentState) -> AgentState:

    joke = state['joke']

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're Assistant, your job is to tell explain the joke."),
        ("user",  "heres a joke {joke} give explanation based on joke" ) 
    ])

    chain =  prompt | llm | StrOutputParser()

    response = chain.invoke({
        "joke" : joke
    })

    return {"explanation" : response}


graph =  StateGraph(AgentState)

graph.add_node("llm", llm_node)
graph.add_node("explanation", explanation)

graph.add_edge(START, "llm")
graph.add_edge("llm" , "explanation")
graph.add_edge("explanation", END)

checkpointer = InMemorySaver()

agent = graph.compile(checkpointer=checkpointer)


config = {"configurable" : {
    "thread_id" : "yogi"}
}

response = agent.invoke({
    "topic":"Artificial Intelligence"
}, config=config)

print(f"\nJoke : {response['joke']}")
print(f"\nExplanation : {response['explanation']}")











    


