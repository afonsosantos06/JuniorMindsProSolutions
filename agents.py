import os
import json
import math
import re
from datetime import datetime
from typing import Any, Dict, Optional

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler

from config import (
    LANGFUSE_ENABLED,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    MODEL_ID,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)


MAX_TEXT_LEN = 260
MAX_MAIL_ITEMS = 2
MAX_SMS_ITEMS = 3
MAX_LOCATION_ITEMS = 6
MAX_LOCATION_SPEED_KMH = 350.0


langfuse_handler = None
if LANGFUSE_ENABLED:
    langfuse_handler = CallbackHandler(public_key=LANGFUSE_PUBLIC_KEY or None)


def _safe_parse_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> float:
    try:
        lat1 = math.radians(float(lat1))
        lon1 = math.radians(float(lon1))
        lat2 = math.radians(float(lat2))
        lon2 = math.radians(float(lon2))
    except (TypeError, ValueError):
        return 0.0

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))
    return 6371.0 * c


def _extract_text(response: Dict[str, Any]) -> str:
    messages = response.get("messages", [])
    if not messages:
        return ""
    content = messages[-1].content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict) and "text" in chunk:
                parts.append(str(chunk["text"]))
            else:
                parts.append(str(chunk))
        return "\n".join(parts).strip()
    return str(content).strip()


def _parse_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def _normalize_decision(decision: str) -> str:
    value = (decision or "").strip().upper()
    if value in {"FRAUD", "LEGITIMATE"}:
        return value
    if any(token in value for token in ["NOT FRAUD", "NO FRAUD", "SAFE", "NORMAL", "GENUINE"]):
        return "LEGITIMATE"
    if any(token in value for token in ["FRAUD", "SCAM", "SUSPIC", "RISKY", "MALICIOUS"]):
        return "FRAUD"
    if "LEGIT" in value:
        return "LEGITIMATE"
    return "LEGITIMATE"


def _decision_from_text(text: str) -> str:
    upper = (text or "").upper()
    if any(token in upper for token in ["NOT FRAUD", "NO FRAUD", "LEGITIMATE", "SAFE", "NORMAL", "GENUINE"]):
        return "LEGITIMATE"
    if any(token in upper for token in ["FRAUD", "SCAM", "SUSPIC", "RISK", "MALICIOUS"]):
        return "FRAUD"
    return "LEGITIMATE"


def _normalize_confidence(raw_value: Any) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, value))


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _to_compact_json(value: Any) -> str:
    sanitized = _sanitize_for_json(value)
    return json.dumps(sanitized, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _trim_text(value: Any, max_len: int = MAX_TEXT_LEN) -> str:
    text = str(value or "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _compact_user(user: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(user, dict):
        return {}
    residence = user.get("residence") or {}
    return {
        "first_name": user.get("first_name"),
        "last_name": user.get("last_name"),
        "job": user.get("job"),
        "salary": user.get("salary"),
        "iban": user.get("iban"),
        "residence": {
            "city": residence.get("city"),
            "lat": residence.get("lat"),
            "lng": residence.get("lng"),
        },
        "description": _trim_text(user.get("description"), 220),
    }


def _find_user_by_iban(users: Any, iban: Optional[str]) -> Optional[Dict[str, Any]]:
    if not iban or not isinstance(users, list):
        return None
    for user in users:
        if isinstance(user, dict) and user.get("iban") == iban:
            return user
    return None


def _sample_text_entries(entries: Any, key: str, terms: list[str], max_items: int) -> list[str]:
    if not isinstance(entries, list):
        return []
    picked: list[str] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        raw = str(item.get(key, ""))
        lower = raw.lower()
        if terms and not any(term in lower for term in terms):
            continue
        picked.append(_trim_text(raw, 220))
        if len(picked) >= max_items:
            break
    if picked:
        return picked

    # Fallback: still give the model a tiny sample if no term match was found.
    for item in entries[:max_items]:
        if isinstance(item, dict):
            picked.append(_trim_text(item.get(key, ""), 160))
    return picked


def _extract_links(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"https?://[^\s'\"]+", text, flags=re.IGNORECASE)


def _looks_phishy_link(link: str) -> bool:
    lower = link.lower()
    suspicious_tokens = [
        "verify",
        "secure",
        "login",
        "signin",
        "update",
        "account",
        "payment",
        "confirm",
        "billing",
        "password",
        "auth",
        "support",
    ]
    suspicious_domains = [
        "paypa1",
        "amaz0n",
        "micr0soft",
        "go0gle",
        "secure-",
        "verify-",
    ]
    if any(token in lower for token in suspicious_tokens):
        return True
    if any(token in lower for token in suspicious_domains):
        return True
    if re.search(r"[a-z]-[a-z0-9]{3,}\.(net|info|top|xyz|shop)", lower):
        return True
    return False


def _phishing_message_hits(messages: list[str]) -> Dict[str, Any]:
    hits: list[Dict[str, Any]] = []
    for message in messages:
        links = _extract_links(message)
        risky_links = [link for link in links if _looks_phishy_link(link)]
        if risky_links:
            hits.append({"links": risky_links, "snippet": _trim_text(message, 200)})
    return {"flagged": bool(hits), "hits": hits[:3]}


def _fast_location_move(sender_locations: Any, threshold_kmh: float = MAX_LOCATION_SPEED_KMH) -> Dict[str, Any]:
    if not isinstance(sender_locations, list) or len(sender_locations) < 2:
        return {"flagged": False, "max_speed_kmh": 0.0, "evidence": []}

    points: list[Dict[str, Any]] = []
    for item in sender_locations:
        if not isinstance(item, dict):
            continue
        parsed_time = _safe_parse_time(item.get("timestamp"))
        if parsed_time is None:
            continue
        points.append(
            {
                "timestamp": parsed_time,
                "city": item.get("city"),
                "lat": item.get("lat"),
                "lng": item.get("lng"),
            }
        )

    points.sort(key=lambda item: item["timestamp"])
    evidence: list[Dict[str, Any]] = []
    max_speed = 0.0

    for previous, current in zip(points, points[1:]):
        elapsed_hours = (current["timestamp"] - previous["timestamp"]).total_seconds() / 3600.0
        if elapsed_hours <= 0:
            continue
        distance_km = _haversine_km(previous["lat"], previous["lng"], current["lat"], current["lng"])
        speed_kmh = distance_km / elapsed_hours if elapsed_hours else 0.0
        max_speed = max(max_speed, speed_kmh)
        if speed_kmh >= threshold_kmh:
            evidence.append(
                {
                    "from": previous["city"],
                    "to": current["city"],
                    "hours": round(elapsed_hours, 2),
                    "distance_km": round(distance_km, 2),
                    "speed_kmh": round(speed_kmh, 2),
                }
            )

    return {"flagged": bool(evidence), "max_speed_kmh": round(max_speed, 2), "evidence": evidence[:3]}


def _compact_locations(locations: Any, sender_id: Optional[str]) -> list[Dict[str, Any]]:
    if not isinstance(locations, list):
        return []
    out: list[Dict[str, Any]] = []
    for item in locations:
        if not isinstance(item, dict):
            continue
        if sender_id and item.get("biotag") != sender_id:
            continue
        out.append(
            {
                "timestamp": item.get("timestamp"),
                "city": item.get("city"),
                "lat": item.get("lat"),
                "lng": item.get("lng"),
            }
        )
        if len(out) >= MAX_LOCATION_ITEMS:
            break
    return out


def _compact_metadata(transaction: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    metadata = metadata or {}
    users = metadata.get("users")
    sender_iban = transaction.get("sender_iban")
    recipient_iban = transaction.get("recipient_iban")
    sender_id = transaction.get("sender_id")
    recipient_id = transaction.get("recipient_id")

    sender_user = _find_user_by_iban(users, sender_iban)
    recipient_user = _find_user_by_iban(users, recipient_iban)

    terms = [
        str(sender_id or "").lower(),
        str(recipient_id or "").lower(),
    ]

    if sender_user:
        terms.extend(
            [
                str(sender_user.get("first_name", "")).lower(),
                str(sender_user.get("last_name", "")).lower(),
                str((sender_user.get("residence") or {}).get("city", "")).lower(),
            ]
        )
    terms = [t for t in terms if t]

    return {
        "counts": {
            "users": len(metadata.get("users", []) or []),
            "locations": len(metadata.get("locations", []) or []),
            "mails": len(metadata.get("mails", []) or []),
            "sms": len(metadata.get("sms", []) or []),
        },
        "sender_user": _compact_user(sender_user or {}),
        "recipient_user": _compact_user(recipient_user or {}),
        "sender_locations": _compact_locations(metadata.get("locations"), sender_id),
        "sender_related_sms": _sample_text_entries(metadata.get("sms"), "sms", terms, MAX_SMS_ITEMS),
        "sender_related_mails": _sample_text_entries(metadata.get("mails"), "mail", terms, MAX_MAIL_ITEMS),
    }


def _log_step(transaction_id: Optional[str], message: str) -> None:
    prefix = f"[{transaction_id}]" if transaction_id else "[analysis]"
    print(f"{prefix} {message}")


def _invoke_config(trace_name: str, transaction_id: Optional[str] = None) -> Dict[str, Any]:
    if not langfuse_handler:
        return {}
    metadata = {"trace_name": trace_name}
    if transaction_id:
        metadata["transaction_id"] = transaction_id
    metadata["langfuse_host"] = LANGFUSE_HOST
    return {
        "callbacks": [langfuse_handler],
        "metadata": metadata,
        "tags": ["fraud-detection", trace_name],
    }


@tool
def check_suspicious_links(message_bundle: str) -> str:
    """Detect phishing-style links in message snippets."""
    try:
        data = json.loads(message_bundle)
    except json.JSONDecodeError:
        data = {"messages": [message_bundle]}

    messages = data.get("messages", [])
    if not isinstance(messages, list):
        messages = [str(messages)]

    result = _phishing_message_hits([str(message) for message in messages])
    return _to_compact_json(result)


@tool
def check_fast_location_moves(location_bundle: str) -> str:
    """Detect impossible travel speeds from location history."""
    try:
        data = json.loads(location_bundle)
    except json.JSONDecodeError:
        data = {"locations": []}

    result = _fast_location_move(data.get("locations", []))
    return _to_compact_json(result)


model = ChatOpenAI(
    api_key=OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY"),
    base_url=OPENROUTER_BASE_URL,
    model=MODEL_ID,
    temperature=0.2,
    max_tokens=1200,
)

triage_agent = create_agent(
    model=model,
    system_prompt=(
        "You are a fraud triage analyst. "
        "Classify each transaction quickly using only the provided context. "
        "Return JSON only with keys: decision, confidence, reason. "
        "Allowed decisions: FRAUD, LEGITIMATE."
    ),
)

context_agent = create_agent(
    model=model,
    system_prompt=(
        "You are a fraud context investigator. "
        "Look for unusual behavior in metadata (users, sms, mails, locations) and transaction details. "
        "Return JSON only with keys: signal_summary, risk_factors, confidence_hint."
    ),
)


@tool
def run_triage(transaction_payload: str) -> str:
    """Run a fast fraud triage on one transaction payload."""
    response = triage_agent.invoke(
        {
            "messages": [
                HumanMessage(content=f"Analyze this transaction payload for fraud: {transaction_payload}")
            ]
        },
        config=_invoke_config("triage_tool"),
    )
    return _extract_text(response)


@tool
def run_context_check(transaction_payload: str) -> str:
    """Run a deeper context check using transaction + metadata."""
    response = context_agent.invoke(
        {
            "messages": [
                HumanMessage(content=f"Investigate context risks for this payload: {transaction_payload}")
            ]
        },
        config=_invoke_config("context_tool"),
    )
    return _extract_text(response)


orchestrator = create_agent(
    model=model,
    system_prompt=(
        "You coordinate a simple fraud multi-agent workflow. "
        "Always call run_triage first. "
        "Call run_context_check if triage confidence is low or if amount looks high/unusual. "
        "Return JSON only with keys: decision, confidence, reason. "
        "Allowed decisions: FRAUD, LEGITIMATE."
    ),
    tools=[run_triage, run_context_check, check_suspicious_links, check_fast_location_moves],
)


def _rule_based_fraud_check(compact_metadata: Dict[str, Any]) -> Dict[str, Any]:
    link_result = _phishing_message_hits(
        [*compact_metadata.get("sender_related_sms", []), *compact_metadata.get("sender_related_mails", [])]
    )
    location_result = _fast_location_move(compact_metadata.get("sender_locations", []))

    if link_result["flagged"]:
        return {
            "flagged": True,
            "decision": "FRAUD",
            "reason": "Suspicious links detected in messages.",
            "evidence": link_result,
        }

    if location_result["flagged"]:
        return {
            "flagged": True,
            "decision": "FRAUD",
            "reason": "Location movement is too fast to be plausible.",
            "evidence": location_result,
        }

    return {"flagged": False}


def analyze_transaction(
    transaction: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Simple public API used by main.py to get a fraud decision for one transaction."""
    transaction_id = str(transaction.get("transaction_id") or "")
    amount = transaction.get("amount")
    sender_id = transaction.get("sender_id")
    recipient_id = transaction.get("recipient_id")

    compact_metadata = _compact_metadata(transaction, metadata)
    _log_step(transaction_id, f"Start analysis: amount={amount}, sender={sender_id}, recipient={recipient_id}")
    if session_id:
        _log_step(transaction_id, f"Langfuse session id: {session_id}")
    _log_step(
        transaction_id,
        "Metadata summary: "
        f"users={compact_metadata['counts']['users']}, "
        f"locations={compact_metadata['counts']['locations']}, "
        f"mails={compact_metadata['counts']['mails']}, "
        f"sms={compact_metadata['counts']['sms']}",
    )

    rule_check = _rule_based_fraud_check(compact_metadata)
    if rule_check.get("flagged"):
        _log_step(transaction_id, f"Rule check triggered FRAUD: {rule_check['reason']}")
        _log_step(transaction_id, f"Rule evidence: {rule_check.get('evidence', {})}")
        return {
            "decision": "FRAUD",
            "confidence": 0.98,
            "reasoning": rule_check["reason"],
            "raw": _to_compact_json(rule_check.get("evidence", {})),
        }

    _log_step(transaction_id, "No hard rule matched. Sending compact payload to the model.")

    payload = {
        "transaction": transaction,
        "metadata": compact_metadata,
    }
    payload_json = _to_compact_json(payload)
    try:
        response = orchestrator.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Analyze this transaction for fraud and return strict JSON only. "
                            f"Payload: {payload_json}"
                        )
                    )
                ]
            },
            config={
                **_invoke_config("orchestrator", str(transaction.get("transaction_id") or "")),
                **({"metadata": {**_invoke_config("orchestrator", str(transaction.get("transaction_id") or "" )).get("metadata", {}), "session_id": session_id}} if session_id else {}),
            },
        )
        _log_step(transaction_id, "Orchestrator call completed.")
    except Exception:
        # Fallback: bypass tool orchestration to avoid invalid/truncated tool-call args.
        _log_step(transaction_id, "Orchestrator failed. Falling back to direct triage.")
        fallback_payload = {
            "transaction": transaction,
            "metadata": {"counts": compact_metadata.get("counts", {})},
        }
        fallback_json = _to_compact_json(fallback_payload)
        response = triage_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Analyze this transaction for fraud. Return strict JSON only with keys: "
                            "decision, confidence, reason. Allowed decisions: FRAUD, LEGITIMATE. "
                            f"Payload: {fallback_json}"
                        )
                    )
                ]
            },
            config={
                **_invoke_config("fallback_triage", str(transaction.get("transaction_id") or "")),
                **({"metadata": {**_invoke_config("fallback_triage", str(transaction.get("transaction_id") or "" )).get("metadata", {}), "session_id": session_id}} if session_id else {}),
            },
        )
        _log_step(transaction_id, "Fallback triage completed.")

    raw_text = _extract_text(response)
    parsed = _parse_json(raw_text)

    parsed_decision = parsed.get("decision")
    if parsed_decision is None:
        parsed_decision = _decision_from_text(raw_text)
    decision = _normalize_decision(str(parsed_decision))
    confidence = _normalize_confidence(parsed.get("confidence", 0.5))
    reason = str(parsed.get("reason", "No reasoning returned by model."))

    _log_step(transaction_id, f"Raw model output: {_trim_text(raw_text, 220)}")
    _log_step(transaction_id, f"Parsed decision={decision}, confidence={confidence:.2f}")
    _log_step(transaction_id, f"Reason: {_trim_text(reason, 220)}")

    return {
        "decision": decision,
        "confidence": confidence,
        "reasoning": reason,
        "raw": raw_text,
    }
