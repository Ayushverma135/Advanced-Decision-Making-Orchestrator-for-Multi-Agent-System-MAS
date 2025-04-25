# descriptions.py
from typing import List, Dict, Any

# --- L1 Supervisor Data ---
l1_route_definitions: List[Dict[str, Any]] = [
    {
        "name": "L2_AuthSupervisor",
        "description": (
            "Handles user access, login, and security credentials. "
            "Focus: Gaining or losing access to the application/service, managing the password."
        ),
        "keywords": ["log in", "login", "signin", "sign in", "authenticate", "access", "log out", "logout", "sign out", "exit", "disconnect", "register", "signup", "sign up", "create account", "new user", "enroll", "password", "forgot", "reset", "change password", "update password", "locked out", "credentials"]
    },
    {
        "name": "L2_AccountSupervisor",
        "description": (
            "Provides information about or modifies details of a specific, existing bank account (requires user to be logged in). "
            "Focus: Information *within* an already authenticated account or changes to account *metadata* (like names associated with it)."
        ),
         "keywords": ["balance", "funds", "money", "history", "transaction", "statement", "activity", "spending", "payments", "deposits", "withdrawals", "details", "number", "account number", "id", "account id", "type", "account type", "update name", "change name", "correct name", "holder name", "legal name", "login name", "display name", "username", "profile name"]
    },
    {
        "name": "L2_SupportSupervisor",
        "description": (
            "Handles general bank information (FAQs), requests for human assistance, and fallback for unclear queries. "
            "Focus: General bank operations, getting help with the service, or when the intent isn't clearly authentication or account data management."
        ),
        "keywords": ["help", "support", "assist", "assistance", "agent", "human", "person", "talk to", "speak to", "representative", "manager", "operator", "issue", "problem", "error", "complain", "complaint", "feedback", "confused", "frustrated", "stuck", "hours", "open", "opening", "closed", "times", "contact", "phone", "call", "email", "location", "address", "branch", "atm", "fees", "charges", "rates", "interest", "website", "app"]
    }
]

# --- L2 Auth Supervisor Data ---
l2_auth_tool_definitions: List[Dict[str, Any]] = [
    {
        "name": "L3_LoginToolNode",
        "description": "Gain access using existing credentials.",
        "keywords": ["log in", "login", "signin", "sign in", "logon", "log on", "authenticate", "access", "enter", "credentials"]
    },
    {
        "name": "L3_SignupToolNode",
        "description": "Create a new account profile.",
        "keywords": ["register", "signup", "sign up", "create", "new account", "open account", "enroll", "join"]
    },
    {
        "name": "L3_PasswordResetToolNode",
        "description": "Reset forgotten password or change existing one.",
        "keywords": ["password", "forgot", "reset", "change", "update", "recover", "locked", "issue", "problem", "help"]
    },
    {
        "name": "L3_LogoutToolNode",
        "description": "End the current active session.",
        "keywords": ["log out", "logout", "sign out", "exit", "leave", "end session", "disconnect", "close"]
    }
]

# --- L2 Account Supervisor Data ---
DISPLAY_DETAILS_ACTION = "L2_DISPLAY_DETAILS" # Special constant for direct handling

l2_account_tool_definitions: List[Dict[str, Any]] = [
    {
        "name": "L3_CheckBalanceToolNode",
        "description": "Check current available funds.",
        "keywords": ["balance", "funds", "how much", "money", "available", "amount"]
    },
    {
        "name": "L3_GetHistoryToolNode",
        "description": "View past transactions and account activity.",
        "keywords": ["history", "transaction", "statement", "activity", "spending", "payments", "deposits", "withdrawals", "log"]
    },
    {
        "name": "L3_AccountHolderNameUpdateToolNode",
        "description": "Update the official/legal name on the account.",
        "keywords": ["holder name", "legal name", "official name", "primary name", "correct name", "fix spelling", "maiden name", "last name"]
    },
    {
        "name": "L3_LoginNameUpdateToolNode",
        "description": "Change the username or display name for login/profile.",
        "keywords": ["login name", "display name", "username", "screen name", "nickname", "profile name", "update name", "alias", "user id"]},
    {
        "name": DISPLAY_DETAILS_ACTION,
        "description": "View account number, type, ID, etc.",
        "keywords": ["details", "number", "account number", "id", "account id", "type", "account type", "info", "summary", "routing", "iban", "sort code"]
    }
]

# --- L2 Support Supervisor Data ---
l2_support_tool_definitions: List[Dict[str, Any]] = [
    {
        "name": "L3_FAQToolNode",
        "description": "Get answers to common questions about bank operations.",
        "keywords": ["hours", "open", "opening", "closed", "times", "holiday", "contact", "phone", "call", "email", "location", "address", "branch", "atm", "fees", "charges", "rates", "interest", "website", "app", "cost", "service"]
    },
    {
        "name": "L3_HumanHandoffNode",
        "description": "Connect with a human agent for complex issues or direct requests.",
        "keywords": ["help", "support", "assist", "agent", "human", "person", "talk to", "speak to", "representative", "manager", "operator", "issue", "problem", "error", "complain", "complaint", "feedback", "confused", "frustrated", "stuck", "security", "fraud", "stolen", "complex", "advice", "escalate", "bypass", "override", "unclear", "ambiguous", "else"]
    }
]

# --- Create global lookup for descriptions ---
# Combine all definitions to create the lookup easily
all_definitions_list = (l1_route_definitions + l2_auth_tool_definitions +
                       l2_account_tool_definitions + l2_support_tool_definitions)
description_lookup: Dict[str, str] = {d["name"]: d["description"] for d in all_definitions_list}

def get_node_description(node_name: str) -> str:
    """Looks up a brief description for a given node name."""
    return description_lookup.get(node_name, "Perform this action") # Default description