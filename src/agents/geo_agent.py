"""GeoAgent: verifies the user was plausibly at the transaction location via GPS pings using strict Impossible Travel math."""
import json
import pandas as pd
from langchain.tools import tool

from src.features import get_nearest_location_pings, haversine_km

@tool
def check_geo(transaction_json: str) -> str:
    """
    Verify user could physically be at transaction location given GPS history.
    Outputs strict Risk Score based on velocity (km/h) between transaction and GPS pings.

    Input: JSON string with keys: transaction_id, sender_id, location (city), timestamp.
    Output: {"risk_score": 0-100, "trigger": "string", "reasoning": "string"}
    Call this: ALWAYS for in-person payments.
    """
    from src.data_loader import get_store
    store = get_store()
    tx = json.loads(transaction_json)
    sender_id = tx.get("sender_id", "")
    tx_location = tx.get("location", "")
    ts_raw = tx.get("timestamp")

    if not tx_location or tx_location.strip() == "":
        return json.dumps({
            "risk_score": 30,
            "trigger": "NO_LOCATION_AVAILABLE",
            "reasoning": "Location data missing for physical validation.",
        })

    pings = store.locations_by_biotag.get(sender_id, [])
    if not pings:
        return json.dumps({
            "risk_score": 40,
            "trigger": "NO_GPS_DATA",
            "reasoning": "Unverified territorial presence (No GPS). Treat with caution.",
        })

    try:
        ts = pd.Timestamp(ts_raw)
    except Exception:
        return json.dumps({
            "risk_score": 30,
            "trigger": "TIMESTAMP_ERROR",
            "reasoning": "Could not parse transaction timestamp.",
        })

    ping_before, ping_after = get_nearest_location_pings(sender_id, ts)
    tx_city = tx_location.split(" - ")[0].strip()

    city_pings = [p for p in pings if p["city"].lower() == tx_city.lower()]
    tx_lat, tx_lng = None, None
    if city_pings:
        nearest_city_ping = min(city_pings, key=lambda p: abs((p["timestamp"] - ts).total_seconds()))
        tx_lat = nearest_city_ping["lat"]
        tx_lng = nearest_city_ping["lng"]

    max_speed_kmh = None

    if ping_before and tx_lat is not None:
        dist_km = haversine_km(ping_before["lat"], ping_before["lng"], tx_lat, tx_lng)
        time_hours = (ts - ping_before["timestamp"]).total_seconds() / 3600
        if time_hours > 0:
            max_speed_kmh = dist_km / time_hours

    if ping_after and tx_lat is not None:
        dist_km = haversine_km(ping_after["lat"], ping_after["lng"], tx_lat, tx_lng)
        time_hours = (ping_after["timestamp"] - ts).total_seconds() / 3600
        if time_hours > 0:
            speed = dist_km / time_hours
            if max_speed_kmh is None or speed > max_speed_kmh:
                max_speed_kmh = speed

    if not ping_before and not ping_after:
        return json.dumps({
            "risk_score": 40,
            "trigger": "NO_RECENT_PINGS",
            "reasoning": "No pings found within 48h. Moderately suspicious.",
        })

    # Hard rules / Risk assignment
    if max_speed_kmh is not None:
        if max_speed_kmh > 800:
            return json.dumps({"risk_score": 100, "trigger": "IMPOSSIBLE_TRAVEL", "reasoning": f"Implied speed {max_speed_kmh:.0f} km/h is physically impossible."})
        elif max_speed_kmh > 300:
            return json.dumps({"risk_score": 80, "trigger": "HIGH_SPEED_TRAVEL", "reasoning": f"Suspicious speed of {max_speed_kmh:.0f} km/h without known plane travel."})
        elif max_speed_kmh > 150:
            return json.dumps({"risk_score": 50, "trigger": "SUSPICIOUS_SPEED", "reasoning": f"Fast transit needed: {max_speed_kmh:.0f} km/h."})
        else:
            return json.dumps({"risk_score": 0, "trigger": "PLAUSIBLE_TRAVEL", "reasoning": f"Normal speed {max_speed_kmh:.0f} km/h."})
    
    # If tx_lat wasn't found, it's a completely unknown city for the user's pings.
    return json.dumps({
        "risk_score": 60,
        "trigger": "UNKNOWN_CITY",
        "reasoning": f"Transaction city {tx_city} has never been visited according to GPS history.",
    })
