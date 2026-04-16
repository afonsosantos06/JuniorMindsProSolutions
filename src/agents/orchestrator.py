"""Orchestrator agent: dynamically executes all tools and sums their rigid risk scores."""
import json
import re

from langchain.agents import create_agent

from src.config import get_model
from src.agents.profile_agent import check_profile
from src.agents.geo_agent import check_geo
from src.agents.behaviour_agent import check_behaviour
from src.agents.comms_agent import check_comms
from src.agents.ml_agent import check_cluster
from src.rules import check_rules

_ORCHESTRATOR_SYSTEM_PROMPT = """You are a mathematical transaction fraud enumerator.
Your goal is not to debate or judge. Your goal is to pass the transaction to EVERY tool, collect their Risk Scores, SUM them up, and output the mathematical result.

REQUIRED WORKFLOW (YOU MUST CALL ALL TOOLS):
1. Call `check_rules`
2. Call `check_cluster`
3. Call `check_behaviour`
4. Call `check_comms`
5. Call `check_profile`
6. Call `check_geo` (only if physical location is available)

FINAL OUTPUT — After collecting the risk scores from all tools, SUM THEM.
If the SUM >= 80, output:
{"transaction_id": "<id>", "verdict": "FRAUD", "confidence": 0.99, "total_risk_score": <sum>}

If the SUM < 80, output:
{"transaction_id": "<id>", "verdict": "LEGIT", "confidence": 0.99, "total_risk_score": <sum>}

Output ONLY the final JSON!"""


# Fields passed to each tool — keep compact to control tokens
_TOOL_FIELDS = [
    "transaction_id", "sender_id", "recipient_id", "transaction_type",
    "amount", "location", "payment_method", "sender_iban", "recipient_iban",
    "balance_after", "description", "timestamp",
]

ALL_TOOLS = [check_rules, check_profile, check_geo, check_behaviour, check_comms, check_cluster]


def _build_agent():
    """Build and return the orchestrator agent graph."""
    model = get_model(temperature=0.1)
    return create_agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=_ORCHESTRATOR_SYSTEM_PROMPT,
    )


# Lazy-init agent so we don't build it at import time
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def _extract_verdict(text: str) -> dict | None:
    """Extract the final JSON verdict from the orchestrator's last message."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Search for embedded JSON with "verdict" key
    m = re.search(r'\{[^{}]*"verdict"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None

def orchestrate_transaction(tx: dict, callback_handler=None, session_id=None) -> tuple:
    """
    Analyze a single transaction and return (transaction_id, is_fraud).

    Args:
        tx: dict with transaction fields
        callback_handler: optional Langfuse CallbackHandler for tracing
        session_id: optional session id for tagging

    Returns:
        (transaction_id: str, is_fraud: bool)
    """
    tx_id = tx.get("transaction_id", "unknown")

    # Build compact JSON for the agent input
    tx_compact = {k: tx.get(k) for k in _TOOL_FIELDS if k in tx}
    # Convert timestamp to string if it's a Timestamp object
    if hasattr(tx_compact.get("timestamp"), "isoformat"):
        tx_compact["timestamp"] = tx_compact["timestamp"].isoformat()

    input_msg = f"Analyze this transaction for fraud:\n{json.dumps(tx_compact)}"

    config = {}
    if callback_handler:
        config["callbacks"] = [callback_handler]
    
    # Langfuse v4 SDK reads session_id from Langchain metadata tag
    if session_id:
        config.setdefault("metadata", {})["langfuse_session_id"] = session_id

    agent = _get_agent()

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": input_msg}]},
            config=config,
        )
        # In LangChain 1.x create_agent returns a state dict with a "messages" list
        messages = result.get("messages", [])
        output = messages[-1].content if messages else ""
        verdict = _extract_verdict(output)

        if verdict:
            is_fraud = str(verdict.get("verdict", "LEGIT")).upper() == "FRAUD"
            # Hard fallback: if explicitly requested to score but it failed, assume it's legit
            
            # Print for debug
            print(f"  [TX {tx_id}] Total Risk: {verdict.get('total_risk_score', 'N/A')} -> {verdict.get('verdict')}")
            return tx_id, is_fraud
        else:
            # Default to LEGIT if we can't parse
            print(f"  [WARN] Could not parse verdict for {tx_id}: {output[:100]}")
            return tx_id, False

    except Exception as e:
        print(f"  [ERROR] Orchestrator failed for {tx_id}: {e}")
        return tx_id, False
