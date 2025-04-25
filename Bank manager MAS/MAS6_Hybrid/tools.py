# tools.py
import random
import string
import uuid
import time
from typing import Dict, Any

# --- Import from common_defs INSTEAD of main_agent ---
from MAS6_hybrid.common_defs import (
    AppState, UserInfo, local_db, faq_db, l3_ambiguity_check,
    get_node_description # get_node_description might not be directly needed here anymore
)
# Import description_lookup if needed (less likely now)
# from descriptions import description_lookup

# Import constants used by tools
from langgraph.graph import END
from langchain_core.messages import HumanMessage
# from descriptions import DISPLAY_DETAILS_ACTION # Only needed if referenced

# --- Authentication Tools ---
def login_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to handle the login process, checks ambiguity first."""
    ambiguity_result = l3_ambiguity_check(state, "L3_LoginToolNode")
    if ambiguity_result: return ambiguity_result

    print("--- Executing Login Tool (Confident) ---")
    email = input("Enter your email: ")
    password = input("Enter your password: ")
    user_data = local_db.get(email)
    result_data: Dict[str, Any] = {}
    if user_data and user_data["password"] == password:
        user_info: UserInfo = {
            "email": email, "name": user_data.get("name", "User"),
            "account_holder_name": user_data.get("account_holder_name", "N/A"),
            "account_number": user_data.get("account_number", "N/A"),
            "account_id": user_data.get("account_id", "N/A"),
            "account_type": user_data.get("account_type", "N/A") }
        result = f"Login successful! Welcome back, {user_info['name']}."
        print(result + "\n")
        result_data = {"user_info": user_info, "task_result": result, "error_message": None}
    else:
        result = "Invalid email or password. Please try again."
        print(result + "\n")
        result_data = {"user_info": None, "task_result": result, "error_message": "Authentication Failed"}

    result_data["top_matches_from_l2"] = None # Clear matches state
    return result_data

def signup_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to handle the signup process, checks ambiguity first."""
    ambiguity_result = l3_ambiguity_check(state, "L3_SignupToolNode")
    if ambiguity_result: return ambiguity_result

    print("--- Executing Signup Tool (Confident) ---")
    email = input("Enter your email: ")
    if email in local_db:
        result = "This email is already registered. Try logging in."
        print(result + "\n")
        return {"task_result": result, "error_message": "Email Exists", "top_matches_from_l2": None}

    login_name = input("Enter your desired login/display name: ")
    account_holder_name = input("Enter the full name for the account holder: ")
    password = input("Enter your password: ")

    if not login_name or not account_holder_name or not password:
         result = "Error: All fields (email, names, password) are required."
         print(result + "\n")
         return {"task_result": result, "error_message": "Missing Signup Field(s)", "top_matches_from_l2": None}

    account_number = ''.join(random.choices(string.digits, k=10)) # Local helper use
    account_id = str(uuid.uuid4()) # Local helper use
    account_type = "Savings"

    local_db[email] = {
        "name": login_name.strip(), "account_holder_name": account_holder_name.strip(), "password": password,
        "balance": 0, "history": [], "account_number": account_number, "account_id": account_id, "account_type": account_type }
    result = f"Sign up successful, {login_name}! Your new {account_type} account ({account_number}) for {account_holder_name} is ready. You can now log in."
    print(result + "\n")
    return {"task_result": result, "error_message": None, "top_matches_from_l2": None}

def password_reset_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to handle password reset, checks ambiguity first."""
    ambiguity_result = l3_ambiguity_check(state, "L3_PasswordResetToolNode")
    if ambiguity_result: return ambiguity_result

    print("--- Executing Password Reset Tool (Confident) ---")
    email = input("Enter the email for the account to reset password: ")
    if email not in local_db:
        result = f"Error: No account found with the email '{email}'."
        print(result + "\n")
        return {"task_result": result, "error_message": "Account Not Found", "top_matches_from_l2": None}

    new_password = input(f"Enter the new password for {email}: ")
    if not new_password or len(new_password) < 3:
        result = "Error: New password is too short. Please try again."
        print(result + "\n")
        return {"task_result": result, "error_message": "Invalid New Password", "top_matches_from_l2": None}

    local_db[email]["password"] = new_password
    result = f"Password for {email} has been updated successfully."
    print(result + "\n")
    return {"task_result": result, "error_message": None, "top_matches_from_l2": None}

def logout_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to handle user logout, checks ambiguity first."""
    ambiguity_result = l3_ambiguity_check(state, "L3_LogoutToolNode")
    if ambiguity_result: return ambiguity_result

    print("--- Executing Logout Tool (Confident) ---")
    result_data: Dict[str, Any] = {}
    if not state.get("user_info"):
        result = "You are not currently logged in."
        result_data = {"task_result": result, "error_message": None}
    else:
        logged_in_name = state["user_info"].get("name", "User")
        result = f"Logging out {logged_in_name}. You have been logged out successfully."
        result_data = {"user_info": None, "task_result": result, "error_message": None} # Clear user_info

    print(result + "\n")
    result_data["top_matches_from_l2"] = None
    return result_data

# --- Account Management Tools ---
def check_balance_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to check balance, checks ambiguity first."""
    ambiguity_result = l3_ambiguity_check(state, "L3_CheckBalanceToolNode")
    if ambiguity_result: return ambiguity_result

    print("--- Executing Check Balance Tool (Confident) ---")
    result_data: Dict[str, Any] = {}
    if not state.get("user_info"):
        result = "Error: You must be logged in to check your balance."
        result_data = {"task_result": result, "error_message": "Authentication Required"}
    else:
        email = state["user_info"]["email"]
        balance = local_db[email].get("balance", None)
        if balance is None:
            result = "Error: Could not retrieve balance information."
            result_data = {"task_result": result, "error_message": "Balance Data Missing"}
        else:
            account_number = state["user_info"].get("account_number", "N/A")
            result = f"Your current balance for account {account_number} is: ${balance:.2f}"
            result_data = {"task_result": result, "error_message": None}

    print(result + "\n")
    result_data["top_matches_from_l2"] = None
    return result_data

def get_history_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to get transaction history, checks ambiguity first."""
    ambiguity_result = l3_ambiguity_check(state, "L3_GetHistoryToolNode")
    if ambiguity_result: return ambiguity_result

    print("--- Executing Get History Tool (Confident) ---")
    result_data: Dict[str, Any] = {}
    if not state.get("user_info"):
        result = "Error: You must be logged in to view transaction history."
        result_data = {"task_result": result, "error_message": "Authentication Required"}
    else:
        email = state["user_info"]["email"]
        history = local_db[email].get("history", [])
        account_number = state["user_info"].get("account_number", "N/A")
        if history:
            history_str = "\n".join([f"- {item}" for item in history])
            result = f"Your recent transactions for account {account_number}:\n{history_str}"
        else:
            result = f"No transaction history found for account {account_number}."
        result_data = {"task_result": result, "error_message": None}

    print(result + "\n")
    result_data["top_matches_from_l2"] = None
    return result_data

def login_name_update_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to update login name, checks ambiguity first."""
    ambiguity_result = l3_ambiguity_check(state, "L3_LoginNameUpdateToolNode")
    if ambiguity_result: return ambiguity_result

    print("--- Executing Login/Display Name Update Tool (Confident) ---")
    if not state.get("user_info"):
        result = "Error: You must be logged in to update your login name."
        print(result + "\n")
        return {"task_result": result, "error_message": "Authentication Required", "top_matches_from_l2": None}

    user_info = state["user_info"]
    email = user_info["email"]
    current_login_name = user_info["name"]
    new_login_name = input(f"Your current login/display name is '{current_login_name}'. Enter the new login name: ")

    if not new_login_name or new_login_name.strip() == "":
        result = "Error: New login name cannot be empty."
        print(result + "\n")
        return {"task_result": result, "error_message": "Invalid New Login Name", "top_matches_from_l2": None}

    new_login_name = new_login_name.strip()
    result_data: Dict[str, Any] = {}
    if email in local_db:
        local_db[email]["name"] = new_login_name
        updated_user_info = user_info.copy()
        updated_user_info["name"] = new_login_name
        result = f"Your login/display name has been updated to '{new_login_name}'."
        result_data = {"task_result": result, "user_info": updated_user_info, "error_message": None}
    else:
        result = "Error: Could not find your account details to update login name."
        result_data = {"task_result": result, "error_message": "Account Data Mismatch", "user_info": None} # Clear user_info on mismatch? Maybe not.

    print(result + "\n")
    result_data["top_matches_from_l2"] = None
    return result_data

def account_holder_name_update_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to update account holder name, checks ambiguity first."""
    ambiguity_result = l3_ambiguity_check(state, "L3_AccountHolderNameUpdateToolNode")
    if ambiguity_result: return ambiguity_result

    print("--- Executing Account Holder Name Update Tool (Confident) ---")
    if not state.get("user_info"):
        result = "Error: You must be logged in to update the account holder name."
        print(result + "\n")
        return {"task_result": result, "error_message": "Authentication Required", "top_matches_from_l2": None}

    user_info = state["user_info"]
    email = user_info["email"]
    current_holder_name = user_info.get("account_holder_name", "N/A")
    new_holder_name = input(f"The current account holder name is '{current_holder_name}'. Enter the new full name for the account holder: ")

    if not new_holder_name or new_holder_name.strip() == "":
        result = "Error: New account holder name cannot be empty."
        print(result + "\n")
        return {"task_result": result, "error_message": "Invalid New Account Holder Name", "top_matches_from_l2": None}

    new_holder_name = new_holder_name.strip()
    result_data: Dict[str, Any] = {}
    if email in local_db:
        local_db[email]["account_holder_name"] = new_holder_name
        updated_user_info = user_info.copy()
        updated_user_info["account_holder_name"] = new_holder_name
        result = f"The account holder name has been updated to '{new_holder_name}'."
        result_data = {"task_result": result, "user_info": updated_user_info, "error_message": None}
    else:
        result = "Error: Could not find your account details to update account holder name."
        result_data = {"task_result": result, "error_message": "Account Data Mismatch", "user_info": None}

    print(result + "\n")
    result_data["top_matches_from_l2"] = None
    return result_data


# --- Support Tools ---
def faq_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to answer FAQs, checks ambiguity first."""
    ambiguity_result = l3_ambiguity_check(state, "L3_FAQToolNode")
    if ambiguity_result: return ambiguity_result

    print("--- Executing FAQ Tool (Confident) ---")
    last_user_message = ""
    task = state.get("current_task", "").lower()
    if not any(kw in task for kw in ["hour", "open", "contact", "phone", "location", "address"]):
         for msg in reversed(state['messages']):
             if isinstance(msg, HumanMessage):
                 last_user_message = msg.content.lower()
                 break
    else:
        last_user_message = task

    result_data: Dict[str, Any] = {}
    result = ""
    if "hour" in last_user_message or "open" in last_user_message:
        result = faq_db.get("hours", "Sorry, I don't have info on hours.")
        result_data = {"task_result": result, "error_message": None}
    elif "contact" in last_user_message or "phone" in last_user_message:
         result = faq_db.get("contact", "Sorry, I don't have contact info.")
         result_data = {"task_result": result, "error_message": None}
    elif "location" in last_user_message or "address" in last_user_message:
         result = faq_db.get("locations", "Sorry, I don't have location info.")
         result_data = {"task_result": result, "error_message": None}
    else:
        print("FAQ tool couldn't find a match based on task/message.")
        result = "I couldn't find a direct answer in the FAQ."
        result_data = {"task_result": result, "error_message": "FAQ Not Found"}

    print(f"FAQ Result: {result}\n")
    result_data["top_matches_from_l2"] = None
    return result_data


def human_handoff_node(state: AppState) -> Dict[str, Any]:
    """Node to initiate human handoff. No ambiguity check needed."""
    print("--- Executing Human Handoff ---")
    result = "Connecting you to a human agent..."
    time.sleep(1)
    print(result + "\n")
    # Clear matches state on handoff
    return {"task_result": result, "next_action": END, "error_message": None, "top_matches_from_l2": None, "suggested_choices": None}