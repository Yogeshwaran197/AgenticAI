import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

load_dotenv()



model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key= os.getenv("GOOGLE_API_KEY")
)

@tool
def Customer_lookup(query: str):
    """Customer service"""
    return f"Customer Record found for query: {query}"


agent = create_agent(
    model=model,
    tools=[Customer_lookup],
    middleware=[
        PIIMiddleware(
            "email",
            strategy= "redact",
            apply_to_input=True
        ),
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True
        )
    ]
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "this is my email - d.yogeshwaran62@gmail.com and this is my card - 4532 0151 1283 0366, can you help me?"
        }
    ]
})

print(result)
