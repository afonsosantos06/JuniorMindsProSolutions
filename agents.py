import os
from typing import TypedDict, Optional, Dict, Any
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe
from langfuse.langchain import CallbackHandler

from config import CHEAP_MODEL_ID, EXPENSIVE_MODEL_ID, OPENROUTER_BASE_URL

# ==========================================
# 🤖 MULTI-AGENT ARCHITECTURE
# This module holds the models and the prompts
# for our two-tier agent approach.
# ==========================================

# --- 1. Graph State Definition ---
# This defines the data that flows between nodes in our LangGraph workflow.
class AgentState(TypedDict):
    transaction: Dict[str, Any]
    session_id: str
    
    # Populated by heuristics
    heuristic_passed: bool
    
    # Populated by agents
    final_decision: Optional[str]      # "FRAUD" or "LEGITIMATE"
    confidence_score: Optional[float]  # 0.0 to 1.0
    reasoning: Optional[str]
    escalated: bool                    # Did the Triage agent escalate to the Deep agent?

# --- 2. Structured Outputs (Pydantic) ---
# We force the LLM to reply exactly in this JSON format.
class TriageResponse(BaseModel):
    decision: str = Field(description="Must be 'LEGITIMATE', 'FRAUD', or 'ESCALATE'")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    reason_spotted: str = Field(description="Short rationale for the decision")

class DeepInvestigatorResponse(BaseModel):
    decision: str = Field(description="Must be exactly 'LEGITIMATE' or 'FRAUD'")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    detailed_chain_of_thought: str = Field(description="Step by step reasoning explaining why this is fraud or not")

# --- 3. Model Initialization ---
# Initialize models using standard parameters.
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

triage_llm = ChatOpenAI(
    api_key=openrouter_api_key,
    base_url=OPENROUTER_BASE_URL,
    model=CHEAP_MODEL_ID,
    temperature=0.1, # Low temperature for more deterministic/stable answers
).with_structured_output(TriageResponse)

investigator_llm = ChatOpenAI(
    api_key=openrouter_api_key,
    base_url=OPENROUTER_BASE_URL,
    model=EXPENSIVE_MODEL_ID,
    temperature=0.3,
).with_structured_output(DeepInvestigatorResponse)


# --- 4. Node Functions (Decorated for Langfuse Tracing) ---

@observe(name="TriageAgentNode")
def call_triage_agent(state: AgentState) -> AgentState:
    """
    The Triage Agent is the first AI wall. It attempts to quickly classify the transaction.
    """
    lf_handler = CallbackHandler()

    prompt = f"""You are a rapid transaction security scanner. 
Analyze the following transaction and determine if it is FRAUD, LEGITIMATE, or if you need to ESCALATE because it is complex or high-value.
Transaction Data: {state["transaction"]}
"""
    messages = [
        SystemMessage(content="You enforce banking security. Be fast and strict."),
        HumanMessage(content=prompt)
    ]
    
    # Invoke model and automatically capture tokens/cost using the Langfuse callback
    response: TriageResponse = triage_llm.invoke(
        messages,
        config={
            "callbacks": [lf_handler],
            "metadata": {
                "langfuse_session_id": state["session_id"],
                "langfuse_trace_name": "TriageAgentNode",
            },
        },
    )
    
    # Update the overall state
    escalated = (response.decision == "ESCALATE")
    
    return {
        **state,
        "final_decision": response.decision if not escalated else None,
        "confidence_score": response.confidence,
        "reasoning": response.reason_spotted,
        "escalated": escalated
    }

@observe(name="DeepInvestigatorNode")
def call_deep_investigator(state: AgentState) -> AgentState:
    """
    The Deep Investigator only runs if the transaction is highly suspicious or very expensive.
    """
    lf_handler = CallbackHandler()

    prompt = f"""You are an elite forensic fraud investigator. 
A transaction has been escalated to you. You MUST make a final, conclusive decision.
Take your time to analyze every single anomaly. 
Transaction Data: {state["transaction"]}
Previous Triage Notes: {state.get("reasoning", "None")}
"""
    messages = [
        SystemMessage(content="You are the ultimate authority on fraud. Analyze deeply and provide a 100% final decision."),
        HumanMessage(content=prompt)
    ]
    
    response: DeepInvestigatorResponse = investigator_llm.invoke(
        messages,
        config={
            "callbacks": [lf_handler],
            "metadata": {
                "langfuse_session_id": state["session_id"],
                "langfuse_trace_name": "DeepInvestigatorNode",
            },
        },
    )
    
    return {
        **state,
        "final_decision": response.decision,
        "confidence_score": response.confidence,
        "reasoning": response.detailed_chain_of_thought,
        "escalated": True # Already escalated
    }
