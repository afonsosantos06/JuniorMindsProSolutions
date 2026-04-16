"""Deterministic rule engine: fast checks with no LLM calls."""
import json
from langchain.tools import tool
import pandas as pd

@tool
def check_rules(transaction_json: str) -> str:
    """
    Run deterministic fraud checks on a transaction. No LLM involved.
    Outputs strict Risk Score.
    
    Input: JSON string with transaction fields.
    Output: {"risk_score": 0-100, "trigger": "string", "reasoning": "string"}
    """
    from src.data_loader import get_store
    tx = json.loads(transaction_json)
    store = get_store()

    triggered = []
    risk = 0

    try:
        amount = float(tx.get("amount", 0))
        balance_after = float(tx.get("balance_after", 0))
        balance_before = balance_after + amount
        if balance_before < 0:
            triggered.append("NEGATIVE_BALANCE_BEFORE")
            risk += 100
        elif amount > balance_before and tx.get("transaction_type") not in ("transfer",):
            triggered.append("AMOUNT_EXCEEDS_BALANCE")
            risk += 100
    except (TypeError, ValueError):
        amount = 0.0

    sender_id = tx.get("sender_id", "")
    recipient_id = tx.get("recipient_id", "")
    if sender_id and recipient_id and sender_id == recipient_id:
        triggered.append("SELF_TRANSFER")
        risk += 100

    sender_iban = str(tx.get("sender_iban", ""))
    if sender_iban and len(sender_iban) >= 2:
        iban_country = sender_iban[:2].upper()
        user = store.users_by_biotag.get(sender_id)
        if user:
            user_iban = str(user.get("iban", ""))
            if user_iban and user_iban[:2].upper() != iban_country:
                triggered.append("IBAN_COUNTRY_MISMATCH")
                risk += 100

    ts_raw = tx.get("timestamp")
    tx_hour = None
    if ts_raw:
        try:
            ts = pd.Timestamp(ts_raw)
            tx_hour = ts.hour
        except Exception:
            pass

    user = store.users_by_biotag.get(sender_id)
    if user and tx_hour is not None and 0 <= tx_hour <= 4:
        from src.features import get_behaviour_profile
        profile = get_behaviour_profile(sender_id)
        if profile and profile.tx_count >= 3:
            if tx_hour < profile.typical_hour_min or tx_hour > profile.typical_hour_max:
                triggered.append("OFF_HOURS_ACTIVITY")
                risk += 60

    if sender_id and recipient_id:
        sender_txs = store.tx_by_sender.get(sender_id)
        if sender_txs is not None and ts_raw:
            try:
                ts = pd.Timestamp(ts_raw)
                prior_recipients = set(sender_txs[sender_txs["timestamp"] < ts]["recipient_id"].dropna())
                if recipient_id not in prior_recipients:
                    from src.features import get_behaviour_profile
                    profile = get_behaviour_profile(sender_id)
                    if profile and profile.amount_mean > 0:
                        if amount > 2 * profile.amount_mean:
                            triggered.append("NEW_RECIPIENT_HIGH_AMOUNT")
                            risk += 70
                    elif not profile or profile.tx_count < 3:
                        if amount > 500:
                            triggered.append("NEW_USER_HIGH_AMOUNT")
                            risk += 80
            except Exception:
                pass

    risk = min(100, risk)
    if risk == 0:
        return json.dumps({"risk_score": 0, "trigger": "CLEAN", "reasoning": "No deterministic rules violated."})
    else:
        return json.dumps({
            "risk_score": risk, 
            "trigger": triggered[0], 
            "reasoning": f"Rules triggered: {', '.join(triggered)}"
        })
