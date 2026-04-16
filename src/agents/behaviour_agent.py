"""BehaviourAgent: detects deviations from the sender's established transaction patterns using pure math scoring."""
import json
import pandas as pd
from langchain.tools import tool

from src.features import get_behaviour_profile

@tool
def check_behaviour(transaction_json: str) -> str:
    """
    Check if this transaction breaks the sender's established behavioral patterns.
    Outputs strict Risk Score based on cumulative heuristics (Z-score amount, hour, new recipient).

    Input: JSON string with keys: transaction_id, sender_id, amount, transaction_type, timestamp, description, recipient_id.
    Output: {"risk_score": 0-100, "trigger": "string", "reasoning": "string"}
    Call this: ALWAYS to analyze deviations.
    """
    from src.data_loader import get_store
    store = get_store()
    tx = json.loads(transaction_json)
    sender_id = tx.get("sender_id", "")

    profile = get_behaviour_profile(sender_id)
    if profile is None or profile.tx_count < 3:
        return json.dumps({
            "risk_score": 60,
            "trigger": "COLD_START_OR_MULE",
            "reasoning": "Insufficient transaction history. Treated as HIGH RISK context.",
        })

    anomalies = []
    risk = 0

    # Hour anomaly
    ts_raw = tx.get("timestamp")
    if ts_raw:
        try:
            tx_hour = pd.Timestamp(ts_raw).hour
            if tx_hour < profile.typical_hour_min or tx_hour > profile.typical_hour_max:
                anomalies.append(f"unusual hour ({tx_hour}:00)")
                risk += 40
        except Exception:
            pass

    # Amount anomaly
    try:
        amount = float(tx.get("amount", 0))
        if profile.amount_std > 0:
            z_score = (amount - profile.amount_mean) / profile.amount_std
            if z_score > 3.0:
                anomalies.append(f"extreme amount (Z={z_score:.1f})")
                risk += 60
            elif z_score > 2.0:
                anomalies.append(f"high amount (Z={z_score:.1f})")
                risk += 45
    except (TypeError, ValueError):
        pass

    # Transaction type
    tx_type = tx.get("transaction_type", "")
    if tx_type and tx_type not in profile.typical_tx_types:
        anomalies.append(f"new tx type ({tx_type})")
        risk += 30

    # Recipient
    recipient_id = tx.get("recipient_id", "")
    if recipient_id and recipient_id not in profile.known_recipients:
        anomalies.append("first-time recipient")
        risk += 45

    risk = min(100, risk)
    
    if risk == 0:
        return json.dumps({
            "risk_score": 0,
            "trigger": "NORMAL_BEHAVIOUR",
            "reasoning": "Matches historical patterns perfectly.",
        })
    elif risk >= 80:
        return json.dumps({
            "risk_score": risk,
            "trigger": "SEVERE_DEVIATION",
            "reasoning": f"Multiple or severe behavioural breaks: {', '.join(anomalies)}",
        })
    else:
        return json.dumps({
            "risk_score": risk,
            "trigger": "MODERATE_DEVIATION",
            "reasoning": f"Deviations found: {', '.join(anomalies)}",
        })
