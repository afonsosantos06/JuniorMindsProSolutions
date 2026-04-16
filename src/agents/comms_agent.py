"""CommsAgent: correlates phishing communications with transactions."""
import json

import pandas as pd
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_model
from src.features import get_phishing_signals

_SYSTEM_PROMPT = """You are a communications-based fraud analyst.
You receive recent phishing attempts targeting a user and a current transaction.
Determine if social engineering may have caused this fraudulent transaction.

Key signals of phishing-induced fraud:
- HIGH-severity phishing (fake bank/PayPal/Amazon security URLs) within 14 days of tx
- Platform match: phishing target matches the transaction platform/description
- User profile described as phishing-susceptible increases risk

Important: Not every transaction after phishing is fraud. Look for platform correlation.
A PayPal phishing SMS followed by a PayPal payment is much more suspicious than
a PayPal phishing SMS followed by a grocery purchase.

Respond ONLY with valid JSON (no extra text):
{"verdict": "FRAUD|LEGIT|UNCERTAIN", "confidence": 0.0-1.0, "reasoning": "one sentence", "matched_phishing": []}"""


@tool
def check_comms(transaction_json: str) -> str:
    """
    Check if sender recently received phishing messages correlated with
    this transaction type or platform.

    Input: JSON string with keys: transaction_id, sender_id, amount,
           transaction_type, description, recipient_id, timestamp.
    Output: JSON string with keys: verdict (FRAUD/LEGIT/UNCERTAIN),
            confidence (0-1), reasoning (1 sentence), matched_phishing (list).
    Call this for: e-commerce transactions, unusual transfers to new recipients,
    PayPal/Amazon/bank-related descriptions, or direct debits at unusual hours.
    """
    from src.data_loader import get_store
    store = get_store()
    tx = json.loads(transaction_json)
    sender_id = tx.get("sender_id", "")

    # Only citizen senders have communications
    if sender_id not in store.users_by_biotag:
        return json.dumps({
            "verdict": "UNCERTAIN",
            "confidence": 0.5,
            "reasoning": "No communications data available for this sender.",
            "matched_phishing": [],
        })

    signals = get_phishing_signals(sender_id)

    if not signals.phishing_events:
        return json.dumps({
            "verdict": "LEGIT",
            "confidence": 0.7,
            "reasoning": "No phishing communications detected for this user.",
            "matched_phishing": [],
        })

    # Filter to events within 30 days before the transaction
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
            "verdict": "LEGIT",
            "confidence": 0.6,
            "reasoning": "No phishing events found within 30 days before this transaction.",
            "matched_phishing": [],
        })

    # Build context for LLM
    user = store.users_by_biotag.get(sender_id, {})
    desc_hint = ""
    if "susceptible" in user.get("description", "").lower() or "confiance" in user.get("description", "").lower():
        desc_hint = " (User is known to be susceptible to phishing.)"

    events_summary = "\n".join(
        f"  - [{e['severity']}] {e['platform']} phishing {e['days_before_tx']}d before tx: {e['text_snippet'][:100]}"
        for e in nearby_events[:5]  # cap at 5 events to control tokens
    )

    context = (
        f"Transaction: {tx.get('transaction_type')}, Amount: {tx.get('amount')}, "
        f"Description: {tx.get('description') or 'N/A'}, Recipient: {tx.get('recipient_id')}{desc_hint}\n\n"
        f"Phishing events within 30 days before this transaction:\n{events_summary}"
    )

    model = get_model(temperature=0.1)
    response = model.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ])

    content = response.content.strip()
    try:
        result = json.loads(content)
        result["matched_phishing"] = nearby_events[:5]
        return json.dumps(result)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{[^{}]+\}', content, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
                result["matched_phishing"] = nearby_events[:5]
                return json.dumps(result)
            except Exception:
                pass
        return json.dumps({
            "verdict": "UNCERTAIN",
            "confidence": 0.6,
            "reasoning": f"Phishing events found but LLM parse failed: {content[:100]}",
            "matched_phishing": nearby_events[:3],
        })
