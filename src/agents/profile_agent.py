"""ProfileAgent: checks if a transaction is consistent with the user's financial profile."""
import json

from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_model
from src.data_loader import get_store

_SYSTEM_PROMPT = """You are a financial profile analyst for fraud detection.
Given a user's salary, job, city, and a transaction, decide if the transaction is plausible for that person.

Key signals of fraud:
- Amount is much larger than expected given monthly salary (monthly salary ≈ annual/12)
- Merchant or service category inconsistent with the user's job and lifestyle
- Transaction description inconsistent with known user habits
- Employer/non-citizen senders (no profile) are treated as UNCERTAIN

Respond ONLY with valid JSON (no extra text):
{"verdict": "FRAUD|LEGIT|UNCERTAIN", "confidence": 0.0-1.0, "reasoning": "one sentence"}"""


@tool
def check_profile(transaction_json: str) -> str:
    """
    Assess if this transaction is consistent with the sender's financial profile.

    Input: JSON string with keys: transaction_id, sender_id, amount,
           transaction_type, location, description, recipient_id.
    Output: JSON string with keys: verdict (FRAUD/LEGIT/UNCERTAIN),
            confidence (0-1), reasoning (1 sentence).
    Call this for: e-commerce transactions, unusually large amounts,
    transactions with merchant categories that may not fit the user's profile.
    """
    store = get_store()
    tx = json.loads(transaction_json)
    sender_id = tx.get("sender_id", "")

    user = store.users_by_biotag.get(sender_id)
    if not user:
        return json.dumps({
            "verdict": "UNCERTAIN",
            "confidence": 0.5,
            "reasoning": "No user profile available for this sender (employer or unknown).",
        })

    # Build compact user context — NOT the full users.json
    monthly_salary = round(user["salary"] / 12, 0)
    user_ctx = (
        f"User: {user['first_name']} {user['last_name']}, "
        f"Job: {user['job']}, "
        f"Annual salary: {user['salary']} (monthly ≈ {monthly_salary}), "
        f"City: {user['residence']['city']}"
    )

    tx_ctx = (
        f"Transaction: {tx.get('transaction_type')}, "
        f"Amount: {tx.get('amount')}, "
        f"Description: {tx.get('description') or 'N/A'}, "
        f"Location: {tx.get('location') or 'N/A'}, "
        f"Recipient: {tx.get('recipient_id')}"
    )

    model = get_model(temperature=0.1)
    response = model.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"{user_ctx}\n{tx_ctx}"),
    ])

    content = response.content.strip()
    # Try to parse; if it fails, return UNCERTAIN
    try:
        result = json.loads(content)
        return json.dumps(result)
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
