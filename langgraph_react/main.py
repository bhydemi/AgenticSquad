from dotenv import load_dotenv
from langchain_core.agents import AgentFinish
from state import AgentState
from langgraph.graph import END, StateGraph
from nodes import run_agent_reasoning_engine, execute_tools

load_dotenv()


AGENT_REASON = "agent_reason"
ACT = "act"


def should_continue(data) -> str:
    if isinstance(data['agent_outcome'], AgentFinish):
        return END
    return ACT


builder = StateGraph(AgentState)
builder.add_node(AGENT_REASON, run_agent_reasoning_engine)
builder.add_node(ACT, execute_tools)
builder.set_entry_point(AGENT_REASON)
builder.add_conditional_edges(AGENT_REASON, should_continue)
builder.add_edge(ACT, AGENT_REASON)
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")



if __name__ == '__main__':
    res = graph.invoke(
        input={
            "input": "what is the weather in Berlin? List it and then Triple it ",
        }
    )
    print(res["agent_outcome"].return_values["output"])