"""BehaviourAgent: detects deviations from the sender's established transaction patterns."""
import json

import pandas as pd
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_model
from src.features import get_behaviour_profile

_SYSTEM_PROMPT = """You are a behavioral anomaly detector for financial transactions.
You receive a summary of a user's typical patterns and the current transaction.
Decide if this transaction is anomalous enough to be suspicious.

Important nuances:
- A single anomaly (e.g., slightly unusual hour for a small recurring amount) is NOT enough for FRAUD.
- Multiple simultaneous anomalies increase suspicion significantly.
- Recurring small direct debits at odd hours may be legitimate scheduled payments.
- If the user has very few prior transactions, patterns are unreliable — lean UNCERTAIN.

Respond ONLY with valid JSON (no extra text):
{"verdict": "FRAUD|LEGIT|UNCERTAIN", "confidence": 0.0-1.0, "reasoning": "one sentence"}"""


@tool
def check_behaviour(transaction_json: str) -> str:
    """
    Check if this transaction breaks the sender's established behavioral patterns.

    Input: JSON string with keys: transaction_id, sender_id, amount,
           transaction_type, timestamp, description, recipient_id.
    Output: JSON string with keys: verdict (FRAUD/LEGIT/UNCERTAIN),
            confidence (0-1), reasoning (1 sentence).
    Call this when: unusual hour, amount far outside normal range,
    transaction type never seen before for this sender, or new recipient.
    """
    from src.data_loader import get_store
    store = get_store()
    tx = json.loads(transaction_json)
    sender_id = tx.get("sender_id", "")

    profile = get_behaviour_profile(sender_id)
    if profile is None or profile.tx_count < 3:
        return json.dumps({
            "verdict": "UNCERTAIN",
            "confidence": 0.5,
            "reasoning": "Insufficient transaction history to build a reliable behaviour profile.",
        })

    # Compute deviations
    anomalies = []

    # Hour anomaly
    ts_raw = tx.get("timestamp")
    tx_hour = None
    if ts_raw:
        try:
            tx_hour = pd.Timestamp(ts_raw).hour
        except Exception:
            pass

    if tx_hour is not None:
        if tx_hour < profile.typical_hour_min or tx_hour > profile.typical_hour_max:
            anomalies.append(f"unusual hour ({tx_hour}:00, typical {profile.typical_hour_min}–{profile.typical_hour_max})")

    # Amount anomaly
    try:
        amount = float(tx.get("amount", 0))
        if profile.amount_std > 0:
            z_score = (amount - profile.amount_mean) / profile.amount_std
            if abs(z_score) > 2.5:
                anomalies.append(
                    f"unusual amount ({amount:.2f}, mean={profile.amount_mean:.2f}, std={profile.amount_std:.2f}, z={z_score:.1f})"
                )
    except (TypeError, ValueError):
        pass

    # Transaction type anomaly
    tx_type = tx.get("transaction_type", "")
    if tx_type and tx_type not in profile.typical_tx_types:
        anomalies.append(f"new transaction type ({tx_type}, usual: {profile.typical_tx_types})")

    # Recipient anomaly
    recipient_id = tx.get("recipient_id", "")
    if recipient_id and recipient_id not in profile.known_recipients:
        anomalies.append(f"first-time recipient ({recipient_id})")

    # Build summary for LLM
    summary = (
        f"Sender: {sender_id}\n"
        f"Typical hours: {profile.typical_hour_min}:00–{profile.typical_hour_max}:00\n"
        f"Typical amount: {profile.amount_mean:.2f} ± {profile.amount_std:.2f}\n"
        f"Typical tx types: {profile.typical_tx_types}\n"
        f"Known recipients: {len(profile.known_recipients)}\n"
        f"Total prior transactions: {profile.tx_count}\n\n"
        f"Current transaction:\n"
        f"  Type: {tx.get('transaction_type')}, Amount: {tx.get('amount')}, "
        f"Hour: {tx_hour}, Recipient: {recipient_id}\n"
        f"  Description: {tx.get('description') or 'N/A'}\n\n"
    )

    if anomalies:
        summary += f"Detected anomalies: {'; '.join(anomalies)}"
    else:
        summary += "No anomalies detected."

    model = get_model(temperature=0.1)
    response = model.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=summary),
    ])

    content = response.content.strip()
    try:
        return json.dumps(json.loads(content))
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{[^{}]+\}', content, re.DOTALL)
        if m:
            try:
                return json.dumps(json.loads(m.group()))
            except Exception:
                pass
        return json.dumps({
            "verdict": "UNCERTAIN",
            "confidence": 0.5,
            "reasoning": f"Could not parse LLM response: {content[:100]}",
        })
