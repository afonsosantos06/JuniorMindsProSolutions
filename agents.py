import os
import json
from typing import TypedDict, Optional, Dict, Any
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe
from langfuse.langchain import CallbackHandler
from langchain_core.tools import tool, render_text_description

from config import CHEAP_MODEL_ID, EXPENSIVE_MODEL_ID, OPENROUTER_BASE_URL
from utils.observability import langfuse_client

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

# --- 4. Tools ---
# Define any tools you want the agents to know about here.

@tool
def get_user_info(iban: str) -> str:
    """Fetch user information such as name, job, and full residence details based on their IBAN (sender_iban or recipient_iban)."""
    try:
        with open("The Truman Show - train/users.json", "r", encoding="utf-8") as f:
            users = json.load(f)
        user = next((u for u in users if u.get("iban") == iban), None)
        return json.dumps(user) if user else f"User with IBAN {iban} not found."
    except Exception as e:
        return f"Error: {e}"

@tool
def get_user_location_history(user_id: str) -> str:
    """Fetch location history for a user based on their user_id (same as sender_id or recipient_id)."""
    try:
        with open("The Truman Show - train/locations.json", "r", encoding="utf-8") as f:
            locations = json.load(f)
        user_locs = [loc for loc in locations if loc.get("biotag") == user_id]
        return json.dumps(user_locs) if user_locs else f"No location history found for {user_id}."
    except Exception as e:
        return f"Error: {e}"

@tool
def get_user_mails(iban: str) -> str:
    """Fetch emails associated with a user based on their IBAN (sender_iban or recipient_iban)."""
    try:
        with open("The Truman Show - train/users.json", "r", encoding="utf-8") as f:
            users = json.load(f)
        user = next((u for u in users if u.get("iban") == iban), None)
        if not user:
            return f"User with IBAN {iban} not found."
        
        first_name = user.get("first_name", "")
        last_name = user.get("last_name", "")
        with open("The Truman Show - train/mails.json", "r", encoding="utf-8") as f:
            mails = json.load(f)
        user_mails = [m for m in mails if first_name in m.get("mail", "") and last_name in m.get("mail", "")]
        return json.dumps(user_mails) if user_mails else f"No emails found for {first_name} {last_name}."
    except Exception as e:
        return f"Error: {e}"

@tool
def get_user_sms(iban: str) -> str:
    """Fetch SMS history associated with a user based on their IBAN (sender_iban or recipient_iban)."""
    try:
        with open("The Truman Show - train/users.json", "r", encoding="utf-8") as f:
            users = json.load(f)
        user = next((u for u in users if u.get("iban") == iban), None)
        if not user:
            return f"User with IBAN {iban} not found."
        
        first_name = user.get("first_name", "")
        with open("The Truman Show - train/sms.json", "r", encoding="utf-8") as f:
            sms_data = json.load(f)
        user_sms = [s for s in sms_data if first_name in s.get("sms", "")]
        return json.dumps(user_sms) if user_sms else f"No SMS found for {first_name}."
    except Exception as e:
        return f"Error: {e}"

TOOLS = [get_user_info, get_user_location_history, get_user_mails, get_user_sms]
tools_description = render_text_description(TOOLS) if TOOLS else ""

# --- 5. Node Functions (Decorated for Langfuse Tracing) ---

@observe(name="TriageAgentNode")
def call_triage_agent(state: AgentState) -> AgentState:
    """
    The Triage Agent is the first AI wall. It attempts to quickly classify the transaction.
    """
    # Link this agent's action to the main dashboard session
    langfuse_client.update_current_trace(session_id=state["session_id"])
    lf_handler = CallbackHandler()

    prompt = f"""You are a rapid transaction security scanner. 
Analyze the following transaction and determine if it is FRAUD, LEGITIMATE, or if you need to ESCALATE because it is complex or high-value.
Transaction Data: {state["transaction"]}
"""
    
    system_content = "You enforce banking security. Be fast and strict."
    if tools_description:
        system_content += f"\n\n--- AVAILABLE TOOLS ---\nYou can use the following tools to gather more information before making a final decision:\n{tools_description}\n\nIMPORTANT: To use a tool, you must explicitly state the tool name and the exact arguments to pass in your 'reason_spotted' field."

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=prompt)
    ]
    
    # Invoke model and automatically capture tokens/cost using the Langfuse callback
    response: TriageResponse = triage_llm.invoke(messages, config={"callbacks": [lf_handler]})
    
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
    langfuse_client.update_current_trace(session_id=state["session_id"])
    lf_handler = CallbackHandler()

    prompt = f"""You are an elite forensic fraud investigator. 
A transaction has been escalated to you. You MUST make a final, conclusive decision.
Take your time to analyze every single anomaly. 
Transaction Data: {state["transaction"]}
Previous Triage Notes: {state.get("reasoning", "None")}
"""
    system_content = "You are the ultimate authority on fraud. Analyze deeply and provide a 100% final decision."
    if tools_description:
        system_content += f"\n\n--- AVAILABLE TOOLS ---\nYou can use the following tools to gather more information before making a final decision:\n{tools_description}\n\nIMPORTANT: To use a tool, you must explicitly state the tool name and the exact arguments to pass in your 'detailed_chain_of_thought' field."

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=prompt)
    ]
    
    response: DeepInvestigatorResponse = investigator_llm.invoke(messages, config={"callbacks": [lf_handler]})
    
    return {
        **state,
        "final_decision": response.decision,
        "confidence_score": response.confidence,
        "reasoning": response.detailed_chain_of_thought,
        "escalated": True # Already escalated
    }
