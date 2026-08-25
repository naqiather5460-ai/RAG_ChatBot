"""
LangGraph orchestration.
Defines the multi-agent graph: a Router node decides which specialist node
handles each question, using LangGraph's conditional routing.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END

from src.rag_chatbot.agents.router_agent import RouterAgent
from src.rag_chatbot.agents.knowledge_agent import KnowledgeAgent
from src.rag_chatbot.agents.directory_agent import DirectoryAgent
from src.rag_chatbot.agents.escalation_agent import EscalationAgent


class GraphState(TypedDict):
    """The shared state passed between nodes in the graph."""
    question: str
    category: str
    answer: str


def build_graph(knowledge_agent: KnowledgeAgent, directory_agent: DirectoryAgent,
                 escalation_agent: EscalationAgent, router_agent: RouterAgent):
    """Builds and compiles the LangGraph multi-agent graph."""

    def router_node(state: GraphState) -> GraphState:
        category = router_agent.route(state["question"])
        return {**state, "category": category}

    def knowledge_node(state: GraphState) -> GraphState:
        answer = knowledge_agent.handle(state["question"])
        return {**state, "answer": answer}

    def directory_node(state: GraphState) -> GraphState:
        answer = directory_agent.handle(state["question"])
        return {**state, "answer": answer}

    def escalation_node(state: GraphState) -> GraphState:
        answer = escalation_agent.handle(state["question"])
        return {**state, "answer": answer}

    def route_decision(state: GraphState) -> str:
        return state["category"]

    graph = StateGraph(GraphState)
    graph.add_node("router", router_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("directory", directory_node)
    graph.add_node("escalation", escalation_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision, {
        "POLICY": "knowledge",
        "DIRECTORY": "directory",
        "ESCALATION": "escalation",
    })
    graph.add_edge("knowledge", END)
    graph.add_edge("directory", END)
    graph.add_edge("escalation", END)

    return graph.compile()