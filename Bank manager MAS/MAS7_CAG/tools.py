# tools.py
import random
import string
import uuid
import time
from typing import Dict, Any

# --- Import from common_defs ---
from common_defs import (
    AppState, UserInfo, local_db, faq_db, check_for_exit
)
# Import Langchain message types if needed
from langchain_core.messages import HumanMessage

# Import constants used by tools
from langgraph.graph import END

# --- Authentication Tools ---
def login_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to handle the login process."""
    # Removed ambiguity check
    print("--- Executing Login Tool ---") # Removed (Confident)

    email = input("Enter your email (or type 'exit' to cancel): ")
    if check_for_exit(email):
        print("Login cancelled.")
        # Return cancellation message, no error
        return {"task_result": "Login cancelled by user.", "error_message": None}

    password = input("Enter your password (or type 'exit' to cancel): ")
    if check_for_exit(password):
        print("Login cancelled.")
        return {"task_result": "Login cancelled by user.", "error_message": None}

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

    # Removed top_matches_from_l2 clearing (no longer passed)
    return result_data

def signup_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to handle the signup process."""
    # Removed ambiguity check
    print("--- Executing Signup Tool ---") # Removed (Confident)

    email = input("Enter your email (or type 'exit' to cancel): ")
    if check_for_exit(email):
        print("Signup cancelled.")
        return {"task_result": "Signup cancelled by user.", "error_message": None}
    if email in local_db:
        result = "This email is already registered. Try logging in."
        print(result + "\n")
        return {"task_result": result, "error_message": "Email Exists"} # Removed top_matches clearing

    login_name = input("Enter your desired login/display name (or type 'exit' to cancel): ")
    if check_for_exit(login_name):
        print("Signup cancelled.")
        return {"task_result": "Signup cancelled by user.", "error_message": None}

    account_holder_name = input("Enter the full name for the account holder (or type 'exit' to cancel): ")
    if check_for_exit(account_holder_name):
        print("Signup cancelled.")
        return {"task_result": "Signup cancelled by user.", "error_message": None}

    password = input("Enter your password (or type 'exit' to cancel): ")
    if check_for_exit(password):
        print("Signup cancelled.")
        return {"task_result": "Signup cancelled by user.", "error_message": None}

    if not login_name or not account_holder_name or not password:
         result = "Error: All fields (email, names, password) are required."
         print(result + "\n")
         return {"task_result": result, "error_message": "Missing Signup Field(s)"} # Removed top_matches clearing

    # Generate account details
    account_number = ''.join(random.choices(string.digits, k=10))
    account_id = str(uuid.uuid4())
    account_type = "Savings"

    local_db[email] = {
        "name": login_name.strip(), "account_holder_name": account_holder_name.strip(), "password": password,
        "balance": 0, "history": [], "account_number": account_number, "account_id": account_id, "account_type": account_type }
    result = f"Sign up successful, {login_name}! Your new {account_type} account ({account_number}) for {account_holder_name} is ready. You can now log in."
    print(result + "\n")
    return {"task_result": result, "error_message": None} # Removed top_matches clearing

def password_reset_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to handle password reset."""
    # Removed ambiguity check
    print("--- Executing Password Reset Tool ---") # Removed (Confident)

    email = input("Enter the email for the account to reset password (or type 'exit' to cancel): ")
    if check_for_exit(email):
        print("Password reset cancelled.")
        return {"task_result": "Password reset cancelled by user.", "error_message": None}
    if email not in local_db:
        result = f"Error: No account found with the email '{email}'."
        print(result + "\n")
        return {"task_result": result, "error_message": "Account Not Found"} # Removed top_matches clearing

    new_password = input(f"Enter the new password for {email} (or type 'exit' to cancel): ")
    if check_for_exit(new_password):
        print("Password reset cancelled.")
        return {"task_result": "Password reset cancelled by user.", "error_message": None}
    if not new_password or len(new_password) < 3:
        result = "Error: New password is too short. Please try again."
        print(result + "\n")
        return {"task_result": result, "error_message": "Invalid New Password"} # Removed top_matches clearing

    local_db[email]["password"] = new_password
    result = f"Password for {email} has been updated successfully."
    print(result + "\n")
    return {"task_result": result, "error_message": None} # Removed top_matches clearing

# Logout doesn't need input, so no change needed
def logout_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to handle user logout."""
    # Removed ambiguity check
    print("--- Executing Logout Tool ---") # Removed (Confident)
    result_data: Dict[str, Any] = {}
    result = ""
    if not state.get("user_info"):
        result = "You are not currently logged in."
        result_data = {"task_result": result, "error_message": None}
    else:
        logged_in_name = state["user_info"].get("name", "User")
        result = f"Logging out {logged_in_name}. You have been logged out successfully."
        result_data = {"user_info": None, "task_result": result, "error_message": None} # Clear user_info

    print(result + "\n")
    # Removed top_matches_from_l2 clearing
    return result_data

# --- Account Management Tools ---
# Balance check doesn't need input
def check_balance_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to check balance."""
    # Removed ambiguity check
    print("--- Executing Check Balance Tool ---") # Removed (Confident)
    result_data: Dict[str, Any] = {}
    result = ""
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
    # Removed top_matches_from_l2 clearing
    return result_data


# History check doesn't need input
def get_history_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to get transaction history."""
    # Removed ambiguity check
    print("--- Executing Get History Tool ---") # Removed (Confident)
    result_data: Dict[str, Any] = {}
    result = ""
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
    # Removed top_matches_from_l2 clearing
    return result_data


def login_name_update_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to update login name."""
    # Removed ambiguity check
    print("--- Executing Login/Display Name Update Tool ---") # Removed (Confident)

    if not state.get("user_info"):
        result = "Error: You must be logged in to update your login name."
        print(result + "\n")
        return {"task_result": result, "error_message": "Authentication Required"} # Removed top_matches clearing

    user_info = state["user_info"]
    email = user_info["email"]
    current_login_name = user_info["name"]

    new_login_name = input(f"Your current login/display name is '{current_login_name}'. Enter the new login name (or type 'exit' to cancel): ")
    if check_for_exit(new_login_name):
        print("Login name update cancelled.")
        return {"task_result": "Login name update cancelled by user.", "error_message": None}

    if not new_login_name or new_login_name.strip() == "":
        result = "Error: New login name cannot be empty."
        print(result + "\n")
        return {"task_result": result, "error_message": "Invalid New Login Name"} # Removed top_matches clearing

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
        result_data = {"task_result": result, "error_message": "Account Data Mismatch"}

    print(result + "\n")
    # Removed top_matches_from_l2 clearing
    return result_data

def account_holder_name_update_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to update account holder name."""
    # Removed ambiguity check
    print("--- Executing Account Holder Name Update Tool ---") # Removed (Confident)

    if not state.get("user_info"):
        result = "Error: You must be logged in to update the account holder name."
        print(result + "\n")
        return {"task_result": result, "error_message": "Authentication Required"} # Removed top_matches clearing

    user_info = state["user_info"]
    email = user_info["email"]
    current_holder_name = user_info.get("account_holder_name", "N/A")

    new_holder_name = input(f"The current account holder name is '{current_holder_name}'. Enter the new full name (or type 'exit' to cancel): ")
    if check_for_exit(new_holder_name):
        print("Account holder name update cancelled.")
        return {"task_result": "Account holder name update cancelled by user.", "error_message": None}

    if not new_holder_name or new_holder_name.strip() == "":
        result = "Error: New account holder name cannot be empty."
        print(result + "\n")
        return {"task_result": result, "error_message": "Invalid New Account Holder Name"} # Removed top_matches clearing

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
        result_data = {"task_result": result, "error_message": "Account Data Mismatch"}

    print(result + "\n")
    # Removed top_matches_from_l2 clearing
    return result_data


# --- Support Tools ---
# FAQ doesn't take input
def faq_tool_node(state: AppState) -> Dict[str, Any]:
    """Node to answer FAQs."""
    # Removed ambiguity check
    print("--- Executing FAQ Tool ---") # Removed (Confident)
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
    # Removed top_matches_from_l2 clearing
    return result_data


# Handoff doesn't take input
def human_handoff_node(state: AppState) -> Dict[str, Any]:
    """Node to initiate human handoff."""
    print("--- Executing Human Handoff ---")
    result = "Connecting you to a human agent..."
    time.sleep(1)
    print(result + "\n")
    # Ensure state cleared
    return {"task_result": result, "next_action": END, "error_message": None, "suggested_choices": None}