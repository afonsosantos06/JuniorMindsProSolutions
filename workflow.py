from typing import Literal
from langgraph.graph import StateGraph, START, END

from agents import AgentState, call_triage_agent, call_deep_investigator
from utils.heuristics import fast_pass_heuristic
from utils.observability import langfuse_client
from config import HIGH_VALUE_THRESHOLD

# ==========================================
# 🗺️ LANGGRAPH WORKFLOW ROUTING
# This controls the logic flow of the Hackathon.
# It decides exactly when to spend budget and 
# when to save it. 
# ==========================================

def evaluate_heuristics_node(state: AgentState) -> AgentState:
    """
    First step: check if we can process this for free (0 token cost).
    """
    transaction = state["transaction"]
    is_resolved, decision, confidence = fast_pass_heuristic(transaction)
    
    # We also check the value here. If it is massively high, we might 
    # want to forcefully ignore the heuristic and go straight to deep investigation.
    amount = float(transaction.get("amount", 0.0))
    if amount >= HIGH_VALUE_THRESHOLD:
        # Override heuristic, this is too risky to skip LLMs
        return {**state, "heuristic_passed": False, "escalated": True}
        
    if is_resolved:
        return {
            **state,
            "heuristic_passed": True,
            "final_decision": decision,
            "confidence_score": confidence,
            "reasoning": "Heuristic fast-pass."
        }
    
    return {**state, "heuristic_passed": False}


def route_after_heuristics(state: AgentState) -> Literal["triage_agent", "deep_investigator", END]: # type: ignore
    """
    Conditional Edge: Decides where to go after the heuristic check.
    """
    if state.get("heuristic_passed"):
        # We got an answer for free! Stop here.
        return END
    
    if state.get("escalated"):
        # It's a high-value transaction, bypass the cheap model.
        return "deep_investigator"
        
    return "triage_agent"


def route_after_triage(state: AgentState) -> Literal["deep_investigator", END]: # type: ignore
    """
    Conditional Edge: Decides where to go after the Triage (cheap) Agent.
    """
    if state.get("escalated"):
        return "deep_investigator"
    
    return END

# --- Build the Graph ---

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("heuristics", evaluate_heuristics_node)
workflow.add_node("triage_agent", call_triage_agent)
workflow.add_node("deep_investigator", call_deep_investigator)

# Define the exact path
workflow.add_edge(START, "heuristics")
workflow.add_conditional_edges("heuristics", route_after_heuristics)
workflow.add_conditional_edges("triage_agent", route_after_triage)
workflow.add_edge("deep_investigator", END)

# Compile into a runnable app
app = workflow.compile()
