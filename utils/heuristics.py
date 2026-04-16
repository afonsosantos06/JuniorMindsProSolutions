from typing import Dict, Any, Tuple
from config import LOW_VALUE_THRESHOLD

# ==========================================
# 🛑 HEURISTICS ENGINE (ZERO COST & ZERO LATENCY)
# This module applies hardcoded business rules.
# If a transaction meets these rules, we don't 
# need to call the AI Agent. This saves our $40 budget
# and decreases our overall system latency (higher score!)
# ==========================================

def fast_pass_heuristic(transaction: Dict[str, Any]) -> Tuple[bool, str, float]:
    """
    Evaluates a transaction against strict rules.
    Returns:
        (is_resolved, final_decision, confidence)
        
        is_resolved: True if the heuristic made a final decision, False if it needs LLM.
        final_decision: "FRAUD" or "LEGITIMATE" or "UNKNOWN"
        confidence: 0.0 to 1.0 representing certainty.
    """
    
    amount = float(transaction.get("amount", 0.0))
    
    # RULE 1: Micro-transactions are rarely targeted for complex fraud, 
    # or the economic penalty of missing them is negligible (Economic Accuracy).
    # We auto-approve these to save budget.
    if amount < LOW_VALUE_THRESHOLD:
        return True, "LEGITIMATE", 0.95
        
    # NOTE FOR THE HACKATHON:
    # Add more rules here based on the dataset attributes!
    # e.g., if transaction.get("ip_country") != transaction.get("card_country"):
    #           return True, "FRAUD", 0.90
    
    # If no rule matches, we must escalate to the LLM Agents
    return False, "UNKNOWN", 0.0
