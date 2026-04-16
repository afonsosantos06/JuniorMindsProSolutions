"""GeoAgent: verifies the user was plausibly at the transaction location via GPS pings."""
import json

import pandas as pd
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_model
from src.features import get_nearest_location_pings, haversine_km

_SYSTEM_PROMPT = """You are a geolocation fraud analyst.
You receive GPS ping data near a transaction timestamp and the required travel speed.
Decide if the user could plausibly be at the transaction location.

Speed thresholds:
- > 900 km/h: physically impossible without supersonic flight → strong FRAUD signal
- 150–900 km/h: suspicious (requires flight) — consider if user is known to travel
- < 150 km/h: plausible by car/train/taxi

Respond ONLY with valid JSON (no extra text):
{"verdict": "FRAUD|LEGIT|UNCERTAIN", "confidence": 0.0-1.0, "reasoning": "one sentence"}"""


@tool
def check_geo(transaction_json: str) -> str:
    """
    Verify user could physically be at transaction location given GPS history.

    Input: JSON string with keys: transaction_id, sender_id, location (city),
           timestamp.
    Output: JSON string with keys: verdict (FRAUD/LEGIT/UNCERTAIN),
            confidence (0-1), reasoning (1 sentence).
    Call this for: in-person payments, or when transaction city differs from home city.
    """
    from src.data_loader import get_store
    store = get_store()
    tx = json.loads(transaction_json)
    sender_id = tx.get("sender_id", "")
    tx_location = tx.get("location", "")
    ts_raw = tx.get("timestamp")

    # If no location, skip
    if not tx_location or tx_location.strip() == "":
        return json.dumps({
            "verdict": "UNCERTAIN",
            "confidence": 0.5,
            "reasoning": "No transaction location available for geo check.",
        })

    # If not a citizen sender (no GPS data), skip
    pings = store.locations_by_biotag.get(sender_id, [])
    if not pings:
        return json.dumps({
            "verdict": "UNCERTAIN",
            "confidence": 0.5,
            "reasoning": "No GPS data available for this sender.",
        })

    try:
        ts = pd.Timestamp(ts_raw)
    except Exception:
        return json.dumps({
            "verdict": "UNCERTAIN",
            "confidence": 0.5,
            "reasoning": "Could not parse transaction timestamp.",
        })

    ping_before, ping_after = get_nearest_location_pings(sender_id, ts)

    # Try to look up coordinates for the transaction city
    # Use the most recent ping city coordinates as a rough proxy if exact match not found
    tx_city = tx_location.split(" - ")[0].strip()  # e.g. "Dietzenbach - Dietzenbach Coffee House" → "Dietzenbach"

    # Find a ping in the transaction city if available
    city_pings = [p for p in pings if p["city"].lower() == tx_city.lower()]
    tx_lat, tx_lng = None, None
    if city_pings:
        # Use the nearest city ping to the transaction time
        nearest_city_ping = min(city_pings, key=lambda p: abs((p["timestamp"] - ts).total_seconds()))
        tx_lat = nearest_city_ping["lat"]
        tx_lng = nearest_city_ping["lng"]

    # Build context for LLM
    context_parts = [f"Transaction city: {tx_city}", f"Transaction time: {ts}"]

    max_speed_kmh = None

    if ping_before and tx_lat is not None:
        dist_km = haversine_km(ping_before["lat"], ping_before["lng"], tx_lat, tx_lng)
        time_hours = (ts - ping_before["timestamp"]).total_seconds() / 3600
        if time_hours > 0:
            speed = dist_km / time_hours
            context_parts.append(
                f"Nearest ping BEFORE tx: {ping_before['city']} at {ping_before['timestamp']} "
                f"({dist_km:.0f} km away, {time_hours:.1f}h gap → required speed {speed:.0f} km/h)"
            )
            max_speed_kmh = speed
        else:
            context_parts.append(f"Nearest ping BEFORE tx: {ping_before['city']} at {ping_before['timestamp']} (same moment)")

    if ping_after and tx_lat is not None:
        dist_km = haversine_km(ping_after["lat"], ping_after["lng"], tx_lat, tx_lng)
        time_hours = (ping_after["timestamp"] - ts).total_seconds() / 3600
        if time_hours > 0:
            speed = dist_km / time_hours
            context_parts.append(
                f"Nearest ping AFTER tx: {ping_after['city']} at {ping_after['timestamp']} "
                f"({dist_km:.0f} km away, {time_hours:.1f}h gap → required speed {speed:.0f} km/h)"
            )
            if max_speed_kmh is None or speed > max_speed_kmh:
                max_speed_kmh = speed

    if not ping_before and not ping_after:
        return json.dumps({
            "verdict": "UNCERTAIN",
            "confidence": 0.5,
            "reasoning": "No GPS pings found within 48h of this transaction.",
        })

    # Hard rule: impossible speed
    if max_speed_kmh is not None and max_speed_kmh > 900:
        return json.dumps({
            "verdict": "FRAUD",
            "confidence": 0.95,
            "reasoning": f"Required travel speed of {max_speed_kmh:.0f} km/h is physically impossible.",
        })

    # If transaction is in same city as pings, likely legit
    if city_pings and max_speed_kmh is not None and max_speed_kmh < 50:
        return json.dumps({
            "verdict": "LEGIT",
            "confidence": 0.85,
            "reasoning": f"User has GPS pings in {tx_city} and required speed is {max_speed_kmh:.0f} km/h.",
        })

    # LLM synthesis for ambiguous cases
    model = get_model(temperature=0.1)
    context = "\n".join(context_parts)
    response = model.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=context),
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
