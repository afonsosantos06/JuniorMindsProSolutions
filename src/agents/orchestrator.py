"""Orchestrator agent: coordinates specialist tools and emits final fraud/legit verdicts."""
import json
import re

from langchain.agents import create_agent

from src.config import get_model
from src.agents.profile_agent import check_profile
from src.agents.geo_agent import check_geo
from src.agents.behaviour_agent import check_behaviour
from src.agents.comms_agent import check_comms
from src.rules import check_rules

_ORCHESTRATOR_SYSTEM_PROMPT = """You are a fraud detection orchestrator for MirrorPay, a financial institution in Reply Mirror (year 2087).
Your job: for each transaction, decide if it is FRAUDULENT or LEGITIMATE.

REQUIRED WORKFLOW — follow this exactly:

STEP 1: ALWAYS call check_rules first.
  - If it returns is_fraud=true (confidence=1.0), output FRAUD immediately without calling other tools.
  - If it returns is_warning=true, note it and continue to relevant specialists.

STEP 2: Call specialists based on transaction type:
  - "in-person" payment → call check_geo AND check_behaviour
  - "e-commerce" → call check_profile AND check_comms AND check_behaviour
  - "transfer" to a recipient not in the sender's known history → call check_profile AND check_behaviour
  - "direct_debit" at unusual hour (00:00–05:00) → call check_behaviour AND check_comms
  - Any transaction at off-hours (hour 0–4) → always call check_behaviour
  - For routine salary/rent transfers at normal hours: check_rules result is sufficient

STEP 3: Aggregate specialist verdicts:
  - 2 or more specialists return FRAUD with confidence > 0.7 → FRAUD
  - RuleEngine warning + 1 specialist FRAUD → FRAUD
  - Any specialist FRAUD with confidence > 0.85 → FRAUD (strong single signal)
  - Otherwise → LEGIT

ASYMMETRIC COST: Missing a fraud is worse than a false positive.
When uncertain and confidence > 0.5 across specialists, lean toward FRAUD.

FINAL OUTPUT — your very last message MUST contain ONLY this JSON (no extra text):
{"transaction_id": "<id>", "verdict": "FRAUD", "confidence": 0.0-1.0}
OR
{"transaction_id": "<id>", "verdict": "LEGIT", "confidence": 0.0-1.0}"""


# Fields passed to each tool — keep compact to control tokens
_TOOL_FIELDS = [
    "transaction_id", "sender_id", "recipient_id", "transaction_type",
    "amount", "location", "payment_method", "sender_iban", "recipient_iban",
    "balance_after", "description", "timestamp",
]

ALL_TOOLS = [check_rules, check_profile, check_geo, check_behaviour, check_comms]


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


def orchestrate_transaction(tx: dict, callback_handler=None) -> tuple:
    """
    Analyze a single transaction and return (transaction_id, is_fraud).

    Args:
        tx: dict with transaction fields
        callback_handler: optional Langfuse CallbackHandler for tracing

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
            return tx_id, is_fraud
        else:
            # Default to LEGIT if we can't parse
            print(f"  [WARN] Could not parse verdict for {tx_id}: {output[:100]}")
            return tx_id, False

    except Exception as e:
        print(f"  [ERROR] Orchestrator failed for {tx_id}: {e}")
        return tx_id, False
