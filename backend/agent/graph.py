from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import analysis_node, reasoning_node, report_node, retrieval_node


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("retrieval")
    graph.set_entry_point("analysis")
    graph.add_edge("analysis", "reasoning")
    graph.add_edge("reasoning", "report")
    graph.add_edge("report", END)

    return graph.compile()


# compiled once, reused across requests
agent = build_agent()