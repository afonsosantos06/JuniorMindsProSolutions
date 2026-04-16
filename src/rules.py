"""Deterministic rule engine: fast checks with no LLM calls."""
import re
from typing import Optional

from pydantic import BaseModel


# IBAN country prefix to 2-letter country code mapping (common ones)
_IBAN_PREFIX_TO_COUNTRY = {
    "FR": "FR", "DE": "DE", "GB": "GB", "IT": "IT", "ES": "ES",
    "NL": "NL", "BE": "BE", "PT": "PT", "AT": "AT", "CH": "CH",
    "PL": "PL", "SE": "SE", "NO": "NO", "DK": "DK", "FI": "FI",
}

# User residence country inferred from IBAN prefix
_USER_COUNTRY_FROM_IBAN = {
    "FR": "FR", "DE": "DE", "GB": "GB",
}


class RuleVerdict(BaseModel):
    transaction_id: str
    is_fraud: bool
    is_warning: bool      # soft flag — not definitive, escalate to LLM
    triggered_rules: list
    confidence: float
    reasoning: str


def check_rules(transaction_json: str) -> str:
    """
    Run deterministic fraud checks on a transaction.
    No LLM is involved. Returns a JSON string.

    Input: JSON string with transaction fields.
    Output: JSON string with keys: transaction_id, is_fraud (bool),
            is_warning (bool), triggered_rules (list), confidence (0-1),
            reasoning (str).

    Checks performed:
    1. amount > balance_before (balance_before = balance_after + amount)
    2. sender_id == recipient_id (self-transfer)
    3. IBAN country prefix mismatch vs sender's registered country
    4. Off-hours activity (00:00-04:59) for known citizen senders
    5. First-seen recipient with amount > 3x sender's typical amount
    """
    import json
    from src.data_loader import get_store

    tx = json.loads(transaction_json)
    store = get_store()

    tx_id = tx.get("transaction_id", "unknown")
    triggered: list = []
    is_fraud = False
    is_warning = False

    # --- Rule 1: amount > balance_before ---
    try:
        amount = float(tx.get("amount", 0))
        balance_after = float(tx.get("balance_after", 0))
        # balance_before ≈ balance_after + amount (for outgoing tx)
        balance_before = balance_after + amount
        if balance_before < 0:
            triggered.append("NEGATIVE_BALANCE_BEFORE")
            is_fraud = True
        elif amount > balance_before and tx.get("transaction_type") not in ("transfer",):
            # For transfers like salary, balance_before might be valid
            triggered.append("AMOUNT_EXCEEDS_BALANCE")
            is_fraud = True
    except (TypeError, ValueError):
        pass

    # --- Rule 2: self-transfer ---
    sender_id = tx.get("sender_id", "")
    recipient_id = tx.get("recipient_id", "")
    if sender_id and recipient_id and sender_id == recipient_id:
        triggered.append("SELF_TRANSFER")
        is_fraud = True

    # --- Rule 3: IBAN country mismatch ---
    sender_iban = str(tx.get("sender_iban", ""))
    if sender_iban and len(sender_iban) >= 2:
        iban_country = sender_iban[:2].upper()
        user = store.users_by_biotag.get(sender_id)
        if user:
            user_iban = str(user.get("iban", ""))
            if user_iban and user_iban[:2].upper() != iban_country:
                triggered.append("IBAN_COUNTRY_MISMATCH")
                is_fraud = True

    # --- Rule 4: Off-hours activity for citizen senders ---
    import pandas as pd
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
        # Check if this is truly anomalous vs their history
        from src.features import get_behaviour_profile
        profile = get_behaviour_profile(sender_id)
        if profile and profile.tx_count >= 3:
            if tx_hour < profile.typical_hour_min or tx_hour > profile.typical_hour_max:
                triggered.append("OFF_HOURS_ACTIVITY")
                is_warning = True  # soft flag only

    # --- Rule 5: First-seen recipient with high amount ---
    if sender_id and recipient_id:
        known_recipients = store.known_recipients.get(sender_id, set())
        # Exclude the current transaction's recipient from the "known" check
        # by checking only prior transactions
        sender_txs = store.tx_by_sender.get(sender_id)
        if sender_txs is not None and ts_raw:
            try:
                ts = pd.Timestamp(ts_raw)
                prior_recipients = set(
                    sender_txs[sender_txs["timestamp"] < ts]["recipient_id"].dropna()
                )
                if recipient_id not in prior_recipients:
                    from src.features import get_behaviour_profile
                    profile = get_behaviour_profile(sender_id)
                    if profile and profile.amount_mean > 0:
                        if amount > 3 * profile.amount_mean:
                            triggered.append("NEW_RECIPIENT_HIGH_AMOUNT")
                            is_warning = True
            except Exception:
                pass

    # Build confidence
    if is_fraud:
        confidence = 1.0
    elif is_warning:
        confidence = 0.6
    else:
        confidence = 0.0

    reasoning_parts = []
    if triggered:
        reasoning_parts.append(f"Rules triggered: {', '.join(triggered)}")
    else:
        reasoning_parts.append("No rule violations detected")

    verdict = RuleVerdict(
        transaction_id=tx_id,
        is_fraud=is_fraud,
        is_warning=is_warning,
        triggered_rules=triggered,
        confidence=confidence,
        reasoning="; ".join(reasoning_parts),
    )
    return verdict.model_dump_json()
