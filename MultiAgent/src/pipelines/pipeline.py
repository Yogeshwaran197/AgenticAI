
from src.agents.agents import bulid_search_agent, bulid_reader_agent, writer_chain, critic_chain



def run_research_pipeline(topic : str) -> dict:

    print("="*60)
    print("Search Agent Working.....")
    print("="*60)

    state = {}

    search_agent = bulid_search_agent()
    result =  search_agent.invoke({
        "messages" : [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })

    state['search_result'] = result["messages"][-1].content

    print(f"\nSearch Results: \n{state['search_result']}\n")


    print("="*60)
    print("Reader Agent Working.....")
    print("="*60)

    reader_agent = bulid_reader_agent()
    result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_result'][:800]}"
        )]

    })

    state['scarped_content'] = result["messages"][-1].content

    print(f"\nScarped Content: \n{state['scarped_content']}\n")

    print("="*60)
    print("Writer Agent Working.....")
    print("="*60)

    combined_research = (
        f"search results : {state['search_result']}\n"
        f"scarped content : {state['scarped_content']}"
    )

    result = writer_chain.invoke({
        "topic": topic,
        "research" : combined_research
    })

    state["writer_report"] = result

    print(f"\n Writer Report: \n{state['writer_report']}\n")

    print("="*60)
    print("Critque Agent Working.....")
    print("="*60)

    report = state['writer_report']

    result = critic_chain.invoke({
        "report" : report
    })

    state["report"] = result

    print(f"\nFinal Report: \n{state['report']}\n")













