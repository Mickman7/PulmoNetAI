from langgraph.graph import StateGraph, END

from . import nodes as nodes_module
from .state import AgentState
from .nodes import (
    analysis_node,
    reasoning_node,
    report_node,
    retrieval_node,
    grade_retrieval_node,
    rewrite_query_node,
)


def _route_after_grade(state: AgentState) -> str:
    """
    Corrective-RAG loop: if the grader judged guideline_context insufficient
    for this case, rewrite the query and retrieve again -- capped by
    MAX_RETRIEVAL_ATTEMPTS (read off the nodes module, not imported by value,
    so tests can tune it down without monkeypatching a stale copy).
    """
    if state.get("retrieval_sufficient", True):
        return "sufficient"
    if state.get("retrieval_attempts", 1) >= nodes_module.MAX_RETRIEVAL_ATTEMPTS:
        return "sufficient"  # cap hit -- proceed with what we have rather than loop forever
    return "retry"


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("grade_retrieval", grade_retrieval_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "grade_retrieval")
    graph.add_conditional_edges(
        "grade_retrieval",
        _route_after_grade,
        {"sufficient": "analysis", "retry": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieval")
    graph.add_edge("analysis", "reasoning")
    graph.add_edge("reasoning", "report")
    graph.add_edge("report", END)

    return graph.compile()


# compiled once, reused across requests
agent = build_agent()
