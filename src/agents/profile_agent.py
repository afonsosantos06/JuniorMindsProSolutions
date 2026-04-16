"""ProfileAgent: checks if a transaction is consistent with the user's financial profile mathematically."""
import json
from langchain.tools import tool
from src.data_loader import get_store

@tool
def check_profile(transaction_json: str) -> str:
    """
    Assess if this transaction is consistent with the sender's financial profile.
    Outputs strict Risk Score based on Monthly Salary ratio anomalies.

    Input: JSON string with keys: transaction_id, sender_id, amount, transaction_type, location, description, recipient_id.
    Output: {"risk_score": 0-100, "trigger": "string", "reasoning": "string"}
    Call this: ALWAYS to analyze wealth-to-spend ratios.
    """
    store = get_store()
    tx = json.loads(transaction_json)
    sender_id = tx.get("sender_id", "")
    
    # Retrieve amount safely
    try:
        amount = float(tx.get("amount", 0))
    except (TypeError, ValueError):
        amount = 0.0

    user = store.users_by_biotag.get(sender_id)
    if not user:
        # Employers transfer huge amounts for payroll, which is normal.
        return json.dumps({
            "risk_score": 20,
            "trigger": "UNLISTED_SENDER",
            "reasoning": "No user profile available (likely employer). Treated as baseline risk.",
        })

    monthly_salary = round(user.get("salary", 15000) / 12, 0)
    risk = 0
    trigger = "PLAUSIBLE_LIFESTYLE"
    anomalies = []

    # Extremely high ratio
    if monthly_salary > 0:
        spend_ratio = amount / monthly_salary
        if spend_ratio > 0.5:
            risk += 80
            trigger = "CRITICAL_SPEND_RATIO"
            anomalies.append(f"Spends > 50% of monthly salary in one tx ({spend_ratio*100:.0f}%)")
        elif spend_ratio > 0.25:
            risk += 40
            trigger = "HIGH_SPEND_RATIO"
            anomalies.append(f"Spends > 25% of monthly salary in one tx ({spend_ratio*100:.0f}%)")
            
    risk = min(100, risk)
    
    if risk == 0:
        return json.dumps({
            "risk_score": 0,
            "trigger": trigger,
            "reasoning": "Transaction amount is fully aligned with user's financial profile.",
        })
    else:
        return json.dumps({
            "risk_score": risk,
            "trigger": trigger,
            "reasoning": f"Profile deviations detected: {'; '.join(anomalies)}.",
        })
