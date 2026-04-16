"""Cached per-user feature extraction: behaviour profiles, phishing signals, location index."""
import math
import re
from functools import lru_cache
from typing import Optional

import pandas as pd
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic models for structured feature outputs
# ---------------------------------------------------------------------------

class BehaviourProfile(BaseModel):
    sender_id: str
    typical_hour_min: int
    typical_hour_max: int
    amount_mean: float
    amount_std: float
    typical_tx_types: list
    known_recipients: list
    tx_count: int


class PhishingEvent(BaseModel):
    date: str
    platform: str
    text_snippet: str
    severity: str  # HIGH | MEDIUM | LOW


class PhishingSignals(BaseModel):
    sender_id: str
    phishing_events: list  # list of PhishingEvent dicts


class LocationPing(BaseModel):
    timestamp: str
    lat: float
    lng: float
    city: str


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Return great-circle distance in km between two lat/lng points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Feature extraction functions
# (Take sender_id and call get_store() internally to allow lru_cache)
# ---------------------------------------------------------------------------

def get_behaviour_profile(sender_id: str) -> Optional[BehaviourProfile]:
    """Compute rolling behavioural stats over sender's historical transactions."""
    from src.data_loader import get_store
    store = get_store()

    sender_txs = store.tx_by_sender.get(sender_id)
    if sender_txs is None or len(sender_txs) == 0:
        return None

    amounts = sender_txs["amount"].astype(float)
    hours = sender_txs["timestamp"].dt.hour

    # Use percentile range covering ~90% of activity
    hour_min = int(hours.quantile(0.05))
    hour_max = int(hours.quantile(0.95))
    if hour_min == hour_max:
        hour_min = max(0, hour_min - 1)
        hour_max = min(23, hour_max + 1)

    amount_mean = float(amounts.mean())
    amount_std = float(amounts.std()) if len(amounts) > 1 else 0.0

    typical_tx_types = list(sender_txs["transaction_type"].value_counts().index)
    known_recipients = list(sender_txs["recipient_id"].dropna().unique())

    return BehaviourProfile(
        sender_id=sender_id,
        typical_hour_min=hour_min,
        typical_hour_max=hour_max,
        amount_mean=amount_mean,
        amount_std=amount_std,
        typical_tx_types=typical_tx_types,
        known_recipients=known_recipients,
        tx_count=len(sender_txs),
    )


# Phishing keyword patterns
_HIGH_SEVERITY_PATTERNS = [
    r"amaz0n", r"paypa1", r"ub3r", r"g00gle", r"micros0ft",
    r"verify\.com", r"secure.*login", r"account.*lock",
]
_MEDIUM_SEVERITY_PATTERNS = [
    r"URGENT", r"verify", r"suspicious", r"unusual login",
    r"confirm.*identity", r"account.*suspend",
]

_HIGH_RE = re.compile("|".join(_HIGH_SEVERITY_PATTERNS), re.IGNORECASE)
_MEDIUM_RE = re.compile("|".join(_MEDIUM_SEVERITY_PATTERNS), re.IGNORECASE)


def _classify_sms_severity(text: str) -> Optional[str]:
    """Return HIGH, MEDIUM, or None for an SMS text."""
    if _HIGH_RE.search(text):
        return "HIGH"
    if _MEDIUM_RE.search(text):
        return "MEDIUM"
    return None


def _extract_platform(text: str) -> str:
    """Guess the platform being impersonated in a phishing message."""
    lowered = text.lower()
    for platform in ["paypal", "amazon", "uber", "google", "microsoft", "facebook",
                     "instagram", "netflix", "apple", "bank"]:
        if platform in lowered:
            return platform.capitalize()
    return "Unknown"


def _extract_date_from_text(text: str) -> str:
    """Extract the first date in YYYY-MM-DD format from a text."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    return m.group(1) if m else "unknown"


def get_phishing_signals(sender_id: str) -> PhishingSignals:
    """Scan SMS and mail for phishing indicators for this user."""
    from src.data_loader import get_store
    store = get_store()

    events: list = []

    for text in store.sms_by_user.get(sender_id, []):
        severity = _classify_sms_severity(text)
        if severity:
            events.append({
                "date": _extract_date_from_text(text),
                "platform": _extract_platform(text),
                "text_snippet": text[:200],
                "severity": severity,
            })

    for text in store.mails_by_user.get(sender_id, []):
        severity = _classify_sms_severity(text)
        if severity:
            events.append({
                "date": _extract_date_from_text(text),
                "platform": _extract_platform(text),
                "text_snippet": text[:200],
                "severity": severity,
            })

    return PhishingSignals(sender_id=sender_id, phishing_events=events)


def get_nearest_location_pings(
    sender_id: str,
    tx_timestamp: pd.Timestamp,
    window_hours: int = 48,
) -> tuple:
    """
    Return (ping_before, ping_after) — the closest GPS pings bracketing
    the transaction timestamp, within window_hours on each side.
    Either may be None if not available.
    """
    from src.data_loader import get_store
    store = get_store()

    pings = store.locations_by_biotag.get(sender_id, [])
    if not pings:
        return None, None

    window = pd.Timedelta(hours=window_hours)
    before = None
    after = None

    for ping in pings:
        ts = ping["timestamp"]
        if ts <= tx_timestamp and (before is None or ts > before["timestamp"]):
            if tx_timestamp - ts <= window:
                before = ping
        elif ts > tx_timestamp and (after is None or ts < after["timestamp"]):
            if ts - tx_timestamp <= window:
                after = ping

    return before, after


# ---------------------------------------------------------------------------
# ML Feature Extraction
# ---------------------------------------------------------------------------

def build_ml_features(tx: dict) -> list[float]:
    """
    Transforms a single transaction dictionary into a numeric feature vector
    suitable for Scikit-Learn KMeans anomaly clustering.
    Returns: [log_amount, hour_scaled, ratio_mean, is_new_recipient]
    """
    import math
    from src.features import get_behaviour_profile

    # 1. Normalized Amount
    amount = float(tx.get("amount", 0.0))
    # use log1p to compress huge variance
    f_amount = math.log1p(max(0, amount))

    # 2. Extract Hour Mapping
    ts_raw = tx.get("timestamp")
    f_hour = 12.0 # default to noon
    if ts_raw:
        try:
            f_hour = float(pd.Timestamp(ts_raw).hour)
        except Exception:
            pass

    # 3. Ratio to user's mean / New recipient flag
    f_ratio = 1.0
    f_new_recip = 0.0

    sender_id = tx.get("sender_id", "")
    recipient_id = tx.get("recipient_id", "")
    profile = get_behaviour_profile(sender_id)

    if profile:
        if profile.amount_mean > 0:
            # capped ratio to avoid explosion with ML
            f_ratio = min(10.0, amount / profile.amount_mean)
        if recipient_id and recipient_id not in profile.known_recipients:
            f_new_recip = 1.0

    return [f_amount, f_hour, f_ratio, f_new_recip]
