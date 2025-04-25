# common_defs.py
import random
import string
import uuid
from typing import TypedDict, List, Optional, Sequence, Tuple, Dict, Any
from langchain_core.messages import BaseMessage

# Import descriptions needed for prompts
from descriptions import description_lookup, get_node_description # Keep these lookups

# Import constants used by tools
from langgraph.graph import END

# --- State Definition ---
class UserInfo(TypedDict):
    email: str
    name: str
    account_holder_name: str
    account_number: str
    account_id: str
    account_type: str

class AppState(TypedDict):
    messages: Sequence[BaseMessage]
    user_info: Optional[UserInfo]
    current_task: Optional[str] # Task description passed down
    task_result: Optional[str]
    next_action: Optional[str] # Node name to go next
    error_message: Optional[str]
    suggested_choices: Optional[List[Tuple[str, str]]] # For clarification node (name, desc)

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

# --- Helper for Exit Check ---
def check_for_exit(user_input: str) -> bool:
    """Checks if user input signifies cancellation."""
    return user_input.strip().lower() in ["exit", "cancel", "quit"]

# --- Clarification Node (MODIFIED Return on Exit) ---
def ask_for_clarification_node(state: AppState) -> dict:
    """Presents choices to the user and gets their selection, allowing exit."""
    print("--- Clarification Needed ---")
    choices = state.get("suggested_choices")
    # Ensure get_node_description is accessible if used
    # from descriptions import get_node_description

    if not choices:
        print("Error: No choices provided for clarification. Routing to support.")
        return {"next_action": "L2_SupportSupervisor", "error_message": "Internal error: Missing clarification choices", "suggested_choices": None}

    print("I'm not sure exactly what you mean. Did you want to:")
    choice_map = {}
    for i, (node_name, description) in enumerate(choices):
        node_desc = get_node_description(node_name) if 'get_node_description' in globals() else description
        label = node_name.split('_')[1] if '_' in node_name else node_name
        print(f"  {i+1}. {label} ({node_desc})")
        choice_map[str(i+1)] = node_name

    while True:
        user_choice_str = input(f"Please enter the number of your choice (1-{len(choices)}) (or type 'exit' to cancel): ")

        # --- EXIT CHECK ---
        if check_for_exit(user_choice_str):
            print("Clarification cancelled.")
            # --- MODIFIED RETURN ---
            # Return task_result to signal cancellation and clear state.
            # Do NOT set next_action. This allows the graph to end the turn.
            return {
                "task_result": "Clarification cancelled by user.",
                "error_message": None,
                "suggested_choices": None,
                "next_action": None # Ensure next_action is None or omitted
            }
        # --- END EXIT CHECK ---

        chosen_node_name = choice_map.get(user_choice_str)
        if chosen_node_name:
            print(f"Okay, proceeding with: {chosen_node_name}")
            # Set next_action for the router, clear choices
            return {"next_action": chosen_node_name, "suggested_choices": None, "error_message": None}
        else:
            try:
                 choice_num = int(user_choice_str)
                 if not (1 <= choice_num <= len(choices)):
                      print(f"Invalid choice. Please enter a number between 1 and {len(choices)} or 'exit'.")
                 else:
                      print("Invalid input. Please enter a valid number or 'exit'.")
            except ValueError:
                 print("Invalid input. Please enter a valid number or 'exit'.")