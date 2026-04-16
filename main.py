import os
import json
import pandas as pd
from typing import List, Dict, Optional, Tuple

from config import TEAM_NAME, DATASET_FOLDER
from utils.observability import generate_session_id
from workflow import app

# ==========================================
# 🚦 HACKATHON ENTRY POINT
# Put this together with your provided datasets.
# Example usage: python main.py
# ==========================================

def load_json_file(filepath: str) -> Optional[Dict]:
    """Load a JSON file safely, return None if file doesn't exist."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ Error loading {filepath}: {e}")
        return None

def load_dataset(dataset_name: str, datasets_dir: str = "datasets") -> Tuple[pd.DataFrame, Dict]:
    """
    Load a complete dataset from the datasets folder structure.
    Returns (transactions_df, metadata_dict) where metadata includes users, locations, mails, sms.
    """
    dataset_path = os.path.join(datasets_dir, dataset_name)
    
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_name} not found.")
        return pd.DataFrame(), {}
    
    # Load transactions
    transactions_file = os.path.join(dataset_path, "transactions.csv")
    if not os.path.exists(transactions_file):
        print(f"❌ No transactions.csv found in {dataset_path}")
        return pd.DataFrame(), {}
    
    transactions_df = pd.read_csv(transactions_file)
    print(f"📊 Loaded {len(transactions_df)} transactions from {dataset_name}")
    
    # Load metadata files
    metadata = {
        "users": load_json_file(os.path.join(dataset_path, "users.json")),
        "locations": load_json_file(os.path.join(dataset_path, "locations.json")),
        "mails": load_json_file(os.path.join(dataset_path, "mails.json")),
        "sms": load_json_file(os.path.join(dataset_path, "sms.json")),
    }
    
    # Log what was loaded
    loaded_files = [k for k, v in metadata.items() if v is not None]
    if loaded_files:
        print(f"📄 Loaded metadata: {', '.join(loaded_files)}")
    
    return transactions_df, metadata

def process_dataset(dataset_name: str) -> None:
    print(f"🚀 Starting processing for {dataset_name}...")
    
    # 1. Generate the crucial Session ID for grading
    session_id = generate_session_id(dataset_name)
    print(f"🆔 Session ID for this run: {session_id}")
    
    # 2. Load the data with metadata
    df, metadata = load_dataset(dataset_name)
    if df.empty:
        print(f"❌ Failed to load dataset {dataset_name}")
        return
    
    flagged_transactions = []
    results = []
    
    # 3. Process row by row
    # (For 6-hour hackathons, synchronous row-by-row is safer for avoiding rate-limits)
    for index, row in df.iterrows():
        transaction = row.to_dict()
        tx_id = transaction.get("transaction_id", f"row_{index}")
        
        print(f"Analyzing {tx_id} (Amount: {transaction.get('amount')})")
        
        # Initial State with detailed transaction structure
        initial_state = {
            "transaction": {
                "transaction_id": row.get("transaction_id"),
                "sender_id": row.get("sender_id"),
                "recipient_id": row.get("recipient_id"),
                "transaction_type": row.get("transaction_type"),
                "amount": row.get("amount"),
                "location": row.get("location"),
                "payment_method": row.get("payment_method"),
                "sender_iban": row.get("sender_iban"),
                "recipient_iban": row.get("recipient_iban"),
                "balance_after": row.get("balance_after"),
                "description": row.get("description"),
                "timestamp": row.get("timestamp")
            },
            "session_id": session_id,
            "heuristic_passed": False,
            "final_decision": None,
            "confidence_score": 0.0,
            "reasoning": "",
            "escalated": False
        }
        
        # Invoke the LangGraph workflow
        final_state = app.invoke(initial_state)
        
        print(f"   -> Decision: {final_state['final_decision']} | Confidence: {final_state['confidence_score']:.2f}")
        
        results.append({
            "transaction_id": tx_id,
            "decision": final_state['final_decision'],
            "confidence": final_state['confidence_score'],
            "reasoning": final_state['reasoning'],
        })
        
        # Track flagged transactions
        if final_state['final_decision'] and final_state['final_decision'].lower() in ['fraud', 'suspicious', 'flagged']:
            flagged_transactions.append(tx_id)
    
    # 4. Save results for the Reply submission portal
    out_file = f"submission_{dataset_name}.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"\n✅ Finished processing {len(df)} transactions. Saved to {out_file}.")
    
    # 5. Save flagged transaction IDs (one per line)
    flagged_file = f"flagged_{dataset_name}.txt"
    with open(flagged_file, 'w') as f:
        for tx_id in flagged_transactions:
            f.write(f"{tx_id}\n")
    print(f"🚩 Flagged {len(flagged_transactions)} transactions. Saved to {flagged_file}.")
    
if __name__ == "__main__":
    # Process the dataset specified in config.py
    process_dataset(DATASET_FOLDER)
