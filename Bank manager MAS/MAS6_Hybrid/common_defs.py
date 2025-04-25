# common_defs.py
import random
import string
import uuid
from typing import TypedDict, List, Optional, Sequence, Annotated, Tuple, Dict, Any

# Import necessary components from descriptions
from MAS6_hybrid.descriptions import (
    l1_route_definitions, l2_auth_tool_definitions, l2_account_tool_definitions,
    l2_support_tool_definitions, description_lookup, get_node_description
)
# You might need to explicitly import BaseMessage if AppState uses it directly
from langchain_core.messages import BaseMessage

# --- State Definition ---
class UserInfo(TypedDict):
    email: str
    name: str
    account_holder_name: str
    account_number: str
    account_id: str
    account_type: str

class AppState(TypedDict):
    messages: Sequence[BaseMessage] # Note: BaseMessage needs importing if not global
    user_info: Optional[UserInfo]
    current_task: Optional[str]
    task_result: Optional[str]
    next_action: Optional[str]
    error_message: Optional[str]
    suggested_choices: Optional[List[Tuple[str, str]]]
    top_matches_from_l2: Optional[List[Tuple[str, float]]]

# --- Databases ---
local_db: Dict[str, Any] = {
    "ayush@gmail.com": {
        "name": "ayush135", "account_holder_name": "Ayush Sharma", "password": "123",
        "balance": 1500.75, "history": ["+ $1000 (Initial Deposit)", "- $50 (Groceries)", "+ $600.75 (Salary)"],
        "account_number": ''.join(random.choices(string.digits, k=10)),
        "account_id": str(uuid.uuid4()), "account_type": "Savings"
    }
}
faq_db: Dict[str, str] = {
    "hours": "Our bank branches are open Mon-Fri 9 AM to 5 PM. Online banking is available 24/7.",
    "contact": "You can call us at 1-800-BANKING or visit our website's contact page.",
    "locations": "We have branches in Pune and Gurugram. Use our online locator for specific addresses."
}

# --- RAG Constants ---
HYBRID_ALPHA = 0.8
DIFF_THRESHOLD = 0.1
MIN_SCORE_THRESHOLD = 0.04

# --- Ambiguity Check Helper ---
# (Depends on DIFF_THRESHOLD and get_node_description which is imported from descriptions)
def l3_ambiguity_check(state: AppState, current_node_name: str) -> Optional[Dict[str, Any]]:
    """Checks for ambiguity based on scores passed from L2.
       Returns state update dict for clarification if ambiguous, else None.
    """
    top_matches = state.get("top_matches_from_l2")
    if not top_matches or len(top_matches) < 2: return None

    if top_matches[0][0] != current_node_name:
        print(f"Warning: [{current_node_name}] Mismatch! Routed here but wasn't top match ({top_matches[0][0]}). Checking ambiguity anyway.")

    top_name, top_score = top_matches[0]
    second_name, second_score = top_matches[1]
    diff = top_score - second_score
    print(f"[{current_node_name}] Ambiguity Check: Top={top_score:.4f}, Second={second_score:.4f}, Diff={diff:.4f}")

    if diff <= DIFF_THRESHOLD:
        print(f"[{current_node_name}] Ambiguity detected (Diff <= {DIFF_THRESHOLD}). Routing to Clarification.")
        choices_to_present = [ (name, get_node_description(name)) for name, score in top_matches[:2] ]
        return {
            "next_action": "AskForClarificationNode",
            "suggested_choices": choices_to_present,
            "top_matches_from_l2": None,
            "error_message": None
        }
    else:
        print(f"[{current_node_name}] High confidence execution (Diff > {DIFF_THRESHOLD}).")
        return None

