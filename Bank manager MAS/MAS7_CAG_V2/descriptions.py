from typing import List, Dict, Any
from constants import (
    L2_AUTH, L2_ACCOUNT, L2_SUPPORT,
    L3_LOGIN, L3_SIGNUP, L3_PWRESET, L3_LOGOUT,
    L3_BALANCE, L3_HISTORY, L3_LOGIN_NAME, L3_HOLDER_NAME,
    L3_FAQ, L3_HANDOFF,
    DISPLAY_DETAILS_ACTION
)

# --- L1 Supervisor Data ---
l1_route_definitions: List[Dict[str, Any]] = [
    {
        "name": L2_AUTH,
        "description": "Manage access: login, logout, signup, and password reset."
    },
    {
        "name": L2_ACCOUNT,
        "description": "Access and update your account details (requires login)."
    },
    {
        "name": L2_SUPPORT,
        "description": "Get bank info, FAQs, or request human support."
    }
]

# --- L2 Auth Supervisor Data ---
l2_auth_tool_definitions: List[Dict[str, Any]] = [
    {"name": L3_LOGIN, "description": "Log in using your credentials."},
    {"name": L3_SIGNUP, "description": "Register a new account."},
    {"name": L3_PWRESET, "description": "Reset or change your password."},
    {"name": L3_LOGOUT, "description": "End your current session."}
]

# --- L2 Account Supervisor Data ---
l2_account_tool_definitions: List[Dict[str, Any]] = [
    {"name": L3_BALANCE, "description": "View your available balance."},
    {"name": L3_HISTORY, "description": "Review your transaction history."},
    {"name": L3_HOLDER_NAME, "description": "Update the account holder's name."},
    {"name": L3_LOGIN_NAME, "description": "Change your login/display name."},
    {"name": DISPLAY_DETAILS_ACTION, "description": "Show complete account details."},
]

# --- L2 Support Supervisor Data ---
l2_support_tool_definitions: List[Dict[str, Any]] = [
    {"name": L3_FAQ, "description": "Find answers to common questions."},
    {"name": L3_HANDOFF, "description": "Connect with a human agent."}
]

# --- Global lookup for descriptions ---
all_definitions_list = (
    l1_route_definitions +
    l2_auth_tool_definitions +
    l2_account_tool_definitions +
    l2_support_tool_definitions
)
description_lookup: Dict[str, str] = {d["name"]: d["description"] for d in all_definitions_list}

def get_node_description(node_name: str) -> str:
    """Look up a concise description for the given node name."""
    return description_lookup.get(node_name, "Perform this action")
