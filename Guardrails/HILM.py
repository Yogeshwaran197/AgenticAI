import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents import create_agent

load_dotenv()

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to} with subject: {subject}"

@tool
def delete_records(table: str, condition: str) -> str:
    """Delete records from the database."""
    return f"Deleted records from {table} where {condition}"



llm = ChatGroq(
    model = "openai/gpt-oss-120b",
)

agent = create_agent(
    model = llm,
    tools = [send_email, delete_records, search_web ],
    middleware= [
        HumanInTheLoopMiddleware(
            interrupt_on= {
                "send_email" : True,
                "delete_records" : True,
                "search_web" : False
            })
    ],
    checkpointer=InMemorySaver()
)


config = {"configurable" : {"thread_id" : "yogi"}}

result = agent.invoke({
    "messages" : [{
        "role":"user",
        "content":"Send an email to team@company.com about the Q4 results"
    }],
}, config=config
)

print(result)

approved = agent.invoke(
    Command(resume= {"decisions": [{"type": "approve"}]}),
    config= config
)
print(approved["messages"][-1].content)
