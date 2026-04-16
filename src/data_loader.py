"""Data loading and indexing: loads all five data files and builds lookup structures."""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_store = None  # module-level singleton


@dataclass
class DataStore:
    transactions: pd.DataFrame
    # iban -> user dict
    users_by_iban: dict
    # sender_id (biotag) -> user dict (only citizen senders, not EMP*)
    users_by_biotag: dict
    # biotag -> list of location pings sorted by timestamp
    locations_by_biotag: dict
    # biotag -> list of SMS texts
    sms_by_user: dict
    # biotag -> list of mail texts
    mails_by_user: dict
    # sender_id -> DataFrame of their transactions sorted by timestamp
    tx_by_sender: dict
    # sender_id -> set of known recipient_ids (from history)
    known_recipients: dict


def _parse_sms_date(sms_text: str):
    """Extract date from SMS text in format 'Date: YYYY-MM-DD HH:MM:SS'."""
    m = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", sms_text)
    if m:
        return m.group(1)
    return None


def _parse_mail_date(mail_text: str):
    """Extract date from mail text."""
    m = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", mail_text)
    if m:
        return m.group(1)
    return None


def load_data(data_dir: str) -> "DataStore":
    """Load all data files from data_dir and return an indexed DataStore."""
    p = Path(data_dir)

    # --- Transactions ---
    transactions = pd.read_csv(p / "transactions.csv")
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
    transactions = transactions.sort_values("timestamp").reset_index(drop=True)

    # --- Users ---
    with open(p / "users.json", encoding="utf-8") as f:
        raw_users = json.load(f)

    users_by_iban = {u["iban"]: u for u in raw_users}

    # Build biotag -> user via sender_iban match
    users_by_biotag: dict = {}
    for _, row in transactions.drop_duplicates("sender_id").iterrows():
        sender_iban = row.get("sender_iban", "")
        user = users_by_iban.get(str(sender_iban))
        if user:
            users_by_biotag[row["sender_id"]] = user

    # --- Locations ---
    with open(p / "locations.json", encoding="utf-8") as f:
        raw_locs = json.load(f)

    locations_by_biotag: dict = {}
    for entry in raw_locs:
        biotag = entry["biotag"]
        locations_by_biotag.setdefault(biotag, []).append({
            "timestamp": pd.Timestamp(entry["timestamp"]),
            "lat": float(entry["lat"]),
            "lng": float(entry["lng"]),
            "city": entry["city"],
        })
    for biotag in locations_by_biotag:
        locations_by_biotag[biotag].sort(key=lambda x: x["timestamp"])

    # --- SMS ---
    with open(p / "sms.json", encoding="utf-8") as f:
        raw_sms = json.load(f)

    # Map first name -> biotag
    name_to_biotag = {u["first_name"]: bt for bt, u in users_by_biotag.items()}
    sms_by_user: dict = {bt: [] for bt in users_by_biotag}

    for entry in raw_sms:
        text = entry.get("sms", "")
        for name, biotag in name_to_biotag.items():
            if name in text:
                sms_by_user[biotag].append(text)
                break

    # --- Mails ---
    with open(p / "mails.json", encoding="utf-8") as f:
        raw_mails = json.load(f)

    mails_by_user: dict = {bt: [] for bt in users_by_biotag}

    for entry in raw_mails:
        # mails.json may be a list of objects with a "mail" or similar key
        if isinstance(entry, dict):
            text = entry.get("mail") or entry.get("email") or entry.get("body") or json.dumps(entry)
        else:
            text = str(entry)
        for name, biotag in name_to_biotag.items():
            if name in text:
                mails_by_user[biotag].append(text)
                break

    # --- Per-sender transaction index ---
    tx_by_sender: dict = {}
    known_recipients: dict = {}
    for sender_id, grp in transactions.groupby("sender_id"):
        tx_by_sender[sender_id] = grp.sort_values("timestamp").reset_index(drop=True)
        known_recipients[sender_id] = set(grp["recipient_id"].dropna().unique())

    return DataStore(
        transactions=transactions,
        users_by_iban=users_by_iban,
        users_by_biotag=users_by_biotag,
        locations_by_biotag=locations_by_biotag,
        sms_by_user=sms_by_user,
        mails_by_user=mails_by_user,
        tx_by_sender=tx_by_sender,
        known_recipients=known_recipients,
    )


def init_store(data_dir: str) -> None:
    """Initialize the module-level DataStore singleton."""
    global _store
    _store = load_data(data_dir)


def get_store() -> DataStore:
    """Return the initialized DataStore singleton."""
    if _store is None:
        raise RuntimeError("DataStore not initialized. Call init_store(data_dir) first.")
    return _store
