"""CommsAgent: correlates phishing communications with transactions securely without LLM bias."""
import json
import pandas as pd
from langchain.tools import tool

from src.features import get_phishing_signals

@tool
def check_comms(transaction_json: str) -> str:
    """
    Check if sender recently received phishing messages correlated with this transaction type or platform.
    Outputs strict Risk Score based on Phishing Severity and recency.
    
    Input: JSON string with keys: transaction_id, sender_id, amount, transaction_type, timestamp.
    Output: {"risk_score": 0-100, "trigger": "string", "reasoning": "string"}
    Call this: ALWAYS for e-commerce, unusual transfers, direct_debit.
    """
    from src.data_loader import get_store
    store = get_store()
    tx = json.loads(transaction_json)
    sender_id = tx.get("sender_id", "")

    if sender_id not in store.users_by_biotag:
        return json.dumps({
            "risk_score": 10,
            "trigger": "CORPORATE_ACCOUNT",
            "reasoning": "Corporate/Employer account. No communications data available.",
        })

    signals = get_phishing_signals(sender_id)

    if not signals.phishing_events:
        return json.dumps({
            "risk_score": 0,
            "trigger": "NO_PHISHING",
            "reasoning": "No phishing communications detected for this user.",
        })

    ts_raw = tx.get("timestamp")
    nearby_events = []
    if ts_raw:
        try:
            tx_ts = pd.Timestamp(ts_raw)
            for event in signals.phishing_events:
                event_date = event.get("date", "unknown")
                if event_date == "unknown":
                    continue
                try:
                    event_ts = pd.Timestamp(event_date)
                    days_before = (tx_ts - event_ts).days
                    if 0 <= days_before <= 30:
                        nearby_events.append({**event, "days_before_tx": days_before})
                except Exception:
                    pass
        except Exception:
            pass

    if not nearby_events:
        return json.dumps({
            "risk_score": 0,
            "trigger": "NO_RECENT_PHISHING",
            "reasoning": "No phishing events found within 30 days before this transaction.",
        })

    # Hard rules / Risk assignment
    max_risk = 0
    trigger = "MILD_PHISHING"
    reasoning = "Phishing detected but low severity or old."

    for e in nearby_events:
        sev = e.get("severity", "LOW")
        days = e.get("days_before_tx", 30)
        
        # Calculate event risk
        event_risk = 0
        if sev == "HIGH":
            if days <= 2:
                event_risk = 95
            elif days <= 14:
                event_risk = 80
            elif days <= 30:
                event_risk = 40
        elif sev == "MEDIUM":
            if days <= 3:
                event_risk = 60
            elif days <= 14:
                event_risk = 40
                
        if event_risk > max_risk:
            max_risk = event_risk
            trigger = f"PHISHING_CORRELATION_{sev}"
            reasoning = f"{sev} severity '{e['platform']}' phishing message received {days} days before transaction."

    # Susceptibility modifier
    user = store.users_by_biotag.get(sender_id, {})
    if "susceptible" in user.get("description", "").lower() or "confiance" in user.get("description", "").lower():
        if max_risk > 0:
            max_risk = min(100, max_risk + 15)
            reasoning += " (User is known to be highly susceptible to phishing)."

    return json.dumps({
        "risk_score": max_risk,
        "trigger": trigger,
        "reasoning": reasoning,
    })
