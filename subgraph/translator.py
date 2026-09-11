import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END , StateGraph
from typing import TypedDict

load_dotenv()


subgraph_llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)

class SubGraph(TypedDict):
    input : str
    translated_text : str


def translator(state: SubGraph) -> SubGraph:

    input = state['input']

    prompt = f"""You're translator, your job is to translate the give input into
        thanglish(tamil+english)
        
        input : {input}

        don't add extra words and don't change the context

        """.strip()

    result = subgraph_llm.invoke(prompt).content

    return {"translated_text" : result}


subgraph_workflow = StateGraph(SubGraph)

subgraph_workflow.add_node("Translator", translator)

subgraph_workflow.add_edge(START, "Translator")
subgraph_workflow.add_edge("Translator", END)

subgraph = subgraph_workflow.compile()


llm = ChatGroq(
    model = "openai/gpt-oss-120b"
)

class ParentGraph(TypedDict):
    question : str
    generation : str
    final_answer :  str


def Generator(state : ParentGraph) -> ParentGraph:

    question = state['question']

    prompt = ChatPromptTemplate.from_template(
        """
        you're helpful AI assistant, you're job is to create an answer for an given
        question {question}

        need answer in plain text no symbols allowed to use
    """
)
    chain = prompt | llm | StrOutputParser()

    result =  chain.invoke(question)

    return {"generation" : result}


def Translator(state : ParentGraph) -> ParentGraph:

    result = subgraph.invoke({
        "input" : state['generation']
    })

    return { "final_answer" : result['translated_text']}


parentgraph_workflow = StateGraph(ParentGraph) 

parentgraph_workflow.add_node("generator", Generator)
parentgraph_workflow.add_node("translator", Translator)

parentgraph_workflow.add_edge(START, "generator")
parentgraph_workflow.add_edge("generator", "translator")
parentgraph_workflow.add_edge("translator", END)


parentgraph = parentgraph_workflow.compile()


print("="*100)
print("ASK ANYTHING")
print("="*100)

while True:
    user_input = input("ASK : ")

    if user_input == "exit":
        break

    result = parentgraph.invoke({
        "question": user_input
    })

    print(f"Generation : {result['generation']}")
    print(f"\nFinal Answer: {result['final_answer']}")