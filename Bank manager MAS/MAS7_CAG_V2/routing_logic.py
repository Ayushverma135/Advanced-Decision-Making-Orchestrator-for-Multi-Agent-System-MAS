# routing_logic.py
from typing import List, Optional, Tuple, Dict, Any

# Import state definition and constants
from state import AppState
from constants import (
    ALL_L2_SUPERVISORS, ALL_L3_TOOLS,
    L2_AUTH, L2_SUPPORT, L3_HANDOFF,
    CLARIFICATION_NODE, END
)
from descriptions import get_node_description # Needed for clarification node

# --- Helper for Exit Check ---
def check_for_exit(user_input: str) -> bool:
    """Checks if user input signifies cancellation."""
    # Ensure check_for_exit is defined here or imported appropriately
    return user_input.strip().lower() in ["exit", "cancel", "quit"]


# --- Clarification Node ---
def ask_for_clarification_node(state: AppState) -> Dict[str, Any]:
    """Presents choices to the user and gets their selection, allowing exit."""
    print("--- Clarification Needed ---")
    choices = state.get("suggested_choices")

    if not choices:
        print("Error: No choices provided for clarification. Routing to support.")
        return {"next_action": L2_SUPPORT, "error_message": "Internal error: Missing clarification choices", "suggested_choices": None}

    print("I'm not sure exactly what you mean. Did you want to:")
    choice_map = {}
    for i, (node_name, description) in enumerate(choices):
        node_desc = get_node_description(node_name) # Use imported helper
        label = node_name.split('_')[1] if '_' in node_name else node_name
        # Clean up label further
        label = label.replace("ToolNode", "").replace("Supervisor", "")
        print(f"  {i+1}. {label} ({node_desc})")
        choice_map[str(i+1)] = node_name

    while True:
        user_choice_str = input(f"Please enter the number of your choice (1-{len(choices)}) (or type 'exit' to cancel): ")

        if check_for_exit(user_choice_str):
            print("Clarification cancelled.")
            # Return task_result, set next_action to None (handled by router)
            return {"task_result": "Clarification cancelled by user.", "error_message": None, "suggested_choices": None, "next_action": None}

        chosen_node_name = choice_map.get(user_choice_str)
        if chosen_node_name:
            print(f"Okay, proceeding with: {chosen_node_name}")
            return {"next_action": chosen_node_name, "suggested_choices": None, "error_message": None}
        else:
            try:
                 choice_num = int(user_choice_str)
                 if not (1 <= choice_num <= len(choices)): print(f"Invalid choice. Please enter a number between 1 and {len(choices)} or 'exit'.")
                 else: print("Invalid input. Please enter a valid number or 'exit'.")
            except ValueError: print("Invalid input. Please enter a valid number or 'exit'.")


# --- Routing Functions ---

def route_l1_decision(state: AppState) -> str:
    """Routes from L1 (LLM based) to L2."""
    next_node = state.get("next_action")
    print(f"[Router] L1 Decision: Route to {next_node}")
    if next_node in ALL_L2_SUPERVISORS: return next_node
    else: print(f"[Router] L1 Warning/Error: Invalid next_action '{next_node}' from L1. Defaulting to Support."); return L2_SUPPORT

def route_l2_decision(state: AppState) -> str:
    """Routes from L2 (LLM based) to L3, Clarification, fallback L2/L3, or END."""
    next_node = state.get("next_action")
    print(f"[Router] L2 Decision: Route to {next_node}")
    valid_targets = ALL_L3_TOOLS + [END, L2_AUTH, L2_SUPPORT, L3_HANDOFF, CLARIFICATION_NODE]
    if next_node in valid_targets:
        return next_node
    else:
        print(f"[Router] L2 Warning/Error: Invalid next_action '{next_node}' from L2. Defaulting to Support.")
        return L2_SUPPORT

def route_after_tool(state: AppState) -> str:
    """Routes after L3 tool execution based on success/error."""
    print(f"[Router] After Tool Execution...")
    error_message = state.get("error_message")
    next_action_forced_by_tool = state.get("next_action")

    if next_action_forced_by_tool == END:
        print("Routing to END (forced by tool).")
        return END

    if error_message:
        print(f"Error detected after tool: {error_message}.")
        if error_message in ["Authentication Required", "Authentication Failed", "Account Data Mismatch"]:
             print("Authentication error detected. Routing to L2 Auth Supervisor.")
             return L2_AUTH
        elif error_message == "FAQ Not Found":
             print("FAQ not found error. Routing to L2 Support Supervisor for handoff.")
             return L2_SUPPORT
        else:
             print("Ending turn due to unhandled tool error.")
             return END
    else:
        # Also includes the case where user cancelled via 'exit' inside the tool
        if state.get("task_result") and "cancelled by user" in state["task_result"]:
            print("Tool operation cancelled by user. Ending turn.")
        else:
            print("Tool executed successfully. Ending current turn.")
        return END

def route_after_clarification(state: AppState) -> str:
    """Routes from Clarification node based on user choice OR cancellation."""
    next_node = state.get("next_action") # Will be L3 tool name or None

    if next_node in ALL_L3_TOOLS: # User made a valid choice
        print(f"[Router] After Clarification: Routing to chosen L3 node: {next_node}")
        return next_node
    elif next_node is None: # User likely cancelled via 'exit'
        print("[Router] After Clarification: No next action set (likely cancelled). Ending turn.")
        return END
    else: # Fallback for invalid next_action set somehow
        print(f"[Router] Clarification Warning/Error: Invalid next_action '{next_node}'. Defaulting to Support.")
        return L2_SUPPORT