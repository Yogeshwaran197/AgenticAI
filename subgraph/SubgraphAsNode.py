import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END , StateGraph
from typing import TypedDict

load_dotenv()

sub_llm = ChatGroq(
    model = "openai/gpt-oss-120b"
)

parent_llm = ChatGroq(
    model = "openai/gpt-oss-120b"
)


class ParentGraph(TypedDict):
    question : str
    generation : str
    final_answer :  str

def translator(state: ParentGraph) -> ParentGraph:

    generation = state['generation']

    prompt = f"""You're translator, your job is to translate the give input into
        thanglish(tamil+english)
        
        input : {generation}

        don't add extra words and don't change the context

        """.strip()

    result = sub_llm.invoke(prompt).content

    return {"final_answer" : result}


sub_workflow = StateGraph(ParentGraph)

sub_workflow.add_node("Translator", translator)

sub_workflow.add_edge(START, "Translator")
sub_workflow.add_edge("Translator", END)

subgraph =  sub_workflow.compile()

def Generator(state : ParentGraph) -> ParentGraph:

    question = state['question']

    prompt = ChatPromptTemplate.from_template(
        """
        you're helpful AI assistant, you're job is to give a accurate answer for an given
        question {question}

        need answer in plain text no symbols allowed to use
    """
)
    chain = prompt | parent_llm | StrOutputParser()

    result =  chain.invoke(question)

    return {"generation" : result}

parent_workflow = StateGraph(ParentGraph)

parent_workflow.add_node("Generator", Generator)
parent_workflow.add_node("Translator", subgraph)

parent_workflow.add_edge(START, "Generator")
parent_workflow.add_edge("Generator", "Translator")
parent_workflow.add_edge("Translator", END)

parentgraph = parent_workflow.compile()

print("="*100)
print("BASIC CHATBOT(THANGLISH)")
print("="*100)

while True:
    user_input = input("ASK : ")

    if user_input == "exit":
        break

    result = parentgraph.invoke({
        "question": user_input
    })
    print(f"\nFinal Answer: {result['final_answer']}")



    