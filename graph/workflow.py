from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from graph.state import AgentState
from agents.nodes import orchestrator_node, researcher_node, critic_node, writer_node

def route_after_critic(state):
    if state["is_sufficient"] or state["revision_count"] >= 3: return "human_review"
    return "researcher"

workflow = StateGraph(AgentState)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("critic", critic_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "orchestrator")
workflow.add_edge("orchestrator", "researcher")
workflow.add_edge("researcher", "critic")
workflow.add_conditional_edges("critic", route_after_critic, {"researcher": "researcher", "human_review": "writer"})
workflow.add_edge("writer", END)

clinical_app = workflow.compile(checkpointer=MemorySaver(), interrupt_before=["writer"])