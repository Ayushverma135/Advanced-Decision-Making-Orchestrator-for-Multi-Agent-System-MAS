import random
import string
import uuid
from typing import TypedDict, List, Optional, Sequence, Annotated
from LLMaas import LLMaaSModel
# Assuming LLMaaSModel is correctly set up in your environment
# from LLMaas import LLMaaSModel
# Placeholder for LLMaaSModel if not available

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
# Removed add_messages import as we handle list directly in main loop now

# Create an instance of LLMaaSModel and get the LLM.
try:
    llmaas_model_instance = LLMaaSModel()
    llm = llmaas_model_instance.get_model()
    # Using MockLLM for demonstration if LLMaaSModel is unavailable
    print(f"Using LLM: {llm.model_name}")
except Exception as e:
    print(f"Fatal Error: Could not initialize LLM model: {e}")
    exit(1)

# -----------------------------------------------------------------------------
# Helper functions (unchanged)
def generate_account_number():
    return ''.join(random.choices(string.digits, k=10))

def generate_account_id():
    return str(uuid.uuid4())

# -----------------------------------------------------------------------------
# Local in-memory database - ADDED account_holder_name
local_db = {
    "ayush@gmail.com": {
        "name": "ayush135",  # Login/display name
        "account_holder_name": "Ayush Sharma", # Official account name
        "password": "123",
        "balance": 1500.75,
        "history": ["+ $1000 (Initial Deposit)", "- $50 (Groceries)", "+ $600.75 (Salary)"],
        "account_number": generate_account_number(),
        "account_id": generate_account_id(),
        "account_type": "Savings"
    }
}

# Simple FAQ database (unchanged)
faq_db = {
    "hours": "Our bank branches are open Mon-Fri 9 AM to 5 PM. Online banking is available 24/7.",
    "contact": "You can call us at 1-800-BANKING or visit our website's contact page.",
    "locations": "We have branches in Pune and Gurugram. Use our online locator for specific addresses."
}

# -----------------------------------------------------------------------------
# State Definition - ADDED account_holder_name to UserInfo
class UserInfo(TypedDict):
    email: str
    name: str                  # Login/display name
    account_holder_name: str   # Official account name
    account_number: str
    account_id: str
    account_type: str

class AppState(TypedDict):
    messages: Sequence[BaseMessage]
    user_info: Optional[UserInfo]
    current_task: Optional[str]
    task_result: Optional[str]
    next_action: Optional[str]
    error_message: Optional[str]

# -----------------------------------------------------------------------------
# Level 3: Specialist Tools / Nodes

# --- Authentication Tools ---

def login_tool_node(state: AppState) -> dict:
    """Node to handle the login process and load user info into state."""
    print("--- Executing Login Tool ---")
    email = input("Enter your email: ")
    password = input("Enter your password: ")
    user_data = local_db.get(email)
    if user_data and user_data["password"] == password:
        # *** UPDATED: Populate full UserInfo including both names ***
        user_info: UserInfo = {
            "email": email,
            "name": user_data.get("name", "User"), # Login name
            "account_holder_name": user_data.get("account_holder_name", "N/A"), # Account holder name
            "account_number": user_data.get("account_number", "N/A"),
            "account_id": user_data.get("account_id", "N/A"),
            "account_type": user_data.get("account_type", "N/A")
        }
        # Use login name for greeting
        result = f"Login successful! Welcome back, {user_info['name']}."
        print(result + "\n")
        return {"user_info": user_info, "task_result": result, "error_message": None}
    else:
        result = "Invalid email or password. Please try again."
        print(result + "\n")
        return {"user_info": None, "task_result": result, "error_message": "Authentication Failed"}

def signup_tool_node(state: AppState) -> dict:
    """Node to handle the signup process."""
    print("--- Executing Signup Tool ---")
    email = input("Enter your email: ")
    if email in local_db:
        result = "This email is already registered. Try logging in."
        print(result + "\n")
        return {"task_result": result, "error_message": "Email Exists"}

    login_name = input("Enter your desired login/display name: ")
    # *** ADDED: Prompt for account holder name ***
    account_holder_name = input("Enter the full name for the account holder: ")
    password = input("Enter your password: ")

    # Basic validation
    if not login_name or not account_holder_name or not password:
         result = "Error: All fields (email, names, password) are required."
         print(result + "\n")
         return {"task_result": result, "error_message": "Missing Signup Field(s)"}

    account_number = generate_account_number()
    account_id = generate_account_id()
    account_type = "Savings" # Default

    local_db[email] = {
        "name": login_name.strip(), # Store login name
        "account_holder_name": account_holder_name.strip(), # Store account holder name
        "password": password,
        "balance": 0,
        "history": [],
        "account_number": account_number,
        "account_id": account_id,
        "account_type": account_type
    }
    result = f"Sign up successful, {login_name}! Your new {account_type} account ({account_number}) for {account_holder_name} is ready. You can now log in."
    print(result + "\n")
    return {"task_result": result, "error_message": None}

def password_reset_tool_node(state: AppState) -> dict:
    """Node to handle password reset (simulated)."""
    print("--- Executing Password Reset Tool ---")
    email = input("Enter the email for the account to reset password: ")
    if email not in local_db:
        result = f"Error: No account found with the email '{email}'."
        print(result + "\n")
        return {"task_result": result, "error_message": "Account Not Found"}

    new_password = input(f"Enter the new password for {email}: ")
    if not new_password or len(new_password) < 3:
        result = "Error: New password is too short. Please try again."
        print(result + "\n")
        return {"task_result": result, "error_message": "Invalid New Password"}

    local_db[email]["password"] = new_password
    result = f"Password for {email} has been updated successfully."
    print(result + "\n")
    return {"task_result": result, "error_message": None}

# --- NEW: Logout Tool ---
def logout_tool_node(state: AppState) -> dict:
    """Node to handle user logout."""
    print("--- Executing Logout Tool ---")
    if not state.get("user_info"):
        result = "You are not currently logged in."
        print(result + "\n")
        # No real error, just informational
        return {"task_result": result, "error_message": None}
    else:
        logged_in_name = state["user_info"].get("name", "User")
        result = f"Logging out {logged_in_name}. You have been logged out successfully."
        print(result + "\n")
        # Clear the user_info from the state
        return {"user_info": None, "task_result": result, "error_message": None}

# --- Account Management Tools ---

def check_balance_tool_node(state: AppState) -> dict:
    # (No changes needed from previous version)
    print("--- Executing Check Balance Tool ---")
    if not state.get("user_info"):
        result = "Error: You must be logged in to check your balance."
        print(result + "\n")
        return {"task_result": result, "error_message": "Authentication Required"}
    email = state["user_info"]["email"]
    balance = local_db[email].get("balance", None)
    if balance is None:
         result = "Error: Could not retrieve balance information."
         print(result + "\n")
         return {"task_result": result, "error_message": "Balance Data Missing"}
    account_number = state["user_info"].get("account_number", "N/A")
    result = f"Your current balance for account {account_number} is: ${balance:.2f}"
    print(result + "\n")
    return {"task_result": result, "error_message": None}

def get_history_tool_node(state: AppState) -> dict:
    # (No changes needed from previous version)
    print("--- Executing Get History Tool ---")
    if not state.get("user_info"):
        result = "Error: You must be logged in to view transaction history."
        print(result + "\n")
        return {"task_result": result, "error_message": "Authentication Required"}
    email = state["user_info"]["email"]
    history = local_db[email].get("history", [])
    account_number = state["user_info"].get("account_number", "N/A")
    if history:
        history_str = "\n".join([f"- {item}" for item in history])
        result = f"Your recent transactions for account {account_number}:\n{history_str}"
    else:
        result = f"No transaction history found for account {account_number}."
    print(result + "\n")
    return {"task_result": result, "error_message": None}

# --- RENAMED: Login Name Update Tool ---
def login_name_update_tool_node(state: AppState) -> dict:
    """Node to update the user's login/display name."""
    print("--- Executing Login/Display Name Update Tool ---")
    if not state.get("user_info"):
        result = "Error: You must be logged in to update your login name."
        print(result + "\n")
        return {"task_result": result, "error_message": "Authentication Required"}

    user_info = state["user_info"]
    email = user_info["email"]
    current_login_name = user_info["name"]
    new_login_name = input(f"Your current login/display name is '{current_login_name}'. Enter the new login name: ")

    if not new_login_name or new_login_name.strip() == "":
        result = "Error: New login name cannot be empty."
        print(result + "\n")
        return {"task_result": result, "error_message": "Invalid New Login Name"}

    new_login_name = new_login_name.strip()

    if email in local_db:
        # Update only the 'name' field in DB
        local_db[email]["name"] = new_login_name
        updated_user_info = user_info.copy()
        updated_user_info["name"] = new_login_name # Update state

        result = f"Your login/display name has been updated to '{new_login_name}'."
        print(result + "\n")
        return {"task_result": result, "user_info": updated_user_info, "error_message": None}
    else:
        result = "Error: Could not find your account details to update login name."
        print(result + "\n")
        return {"task_result": result, "error_message": "Account Data Mismatch", "user_info": None}

# --- NEW: Account Holder Name Update Tool ---
def account_holder_name_update_tool_node(state: AppState) -> dict:
    """Node to update the official account holder name."""
    print("--- Executing Account Holder Name Update Tool ---")
    if not state.get("user_info"):
        result = "Error: You must be logged in to update the account holder name."
        print(result + "\n")
        return {"task_result": result, "error_message": "Authentication Required"}

    user_info = state["user_info"]
    email = user_info["email"]
    current_holder_name = user_info.get("account_holder_name", "N/A") # Get from state
    new_holder_name = input(f"The current account holder name is '{current_holder_name}'. Enter the new full name for the account holder: ")

    if not new_holder_name or new_holder_name.strip() == "":
        result = "Error: New account holder name cannot be empty."
        print(result + "\n")
        return {"task_result": result, "error_message": "Invalid New Account Holder Name"}

    new_holder_name = new_holder_name.strip()

    if email in local_db:
        # Update only the 'account_holder_name' field in DB
        local_db[email]["account_holder_name"] = new_holder_name
        updated_user_info = user_info.copy()
        updated_user_info["account_holder_name"] = new_holder_name # Update state

        result = f"The account holder name has been updated to '{new_holder_name}'."
        print(result + "\n")
        # In a real bank, this might require verification/documentation.
        return {"task_result": result, "user_info": updated_user_info, "error_message": None}
    else:
        result = "Error: Could not find your account details to update account holder name."
        print(result + "\n")
        return {"task_result": result, "error_message": "Account Data Mismatch", "user_info": None}


# --- Support Tools ---

def faq_tool_node(state: AppState) -> dict:
    # (No changes needed from previous version)
    print("--- Executing FAQ Tool ---")
    last_user_message = ""
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content.lower()
            break
    if "hour" in last_user_message or "open" in last_user_message:
        result = faq_db.get("hours", "Sorry, I don't have info on hours.")
    elif "contact" in last_user_message or "phone" in last_user_message:
         result = faq_db.get("contact", "Sorry, I don't have contact info.")
    elif "location" in last_user_message or "address" in last_user_message:
         result = faq_db.get("locations", "Sorry, I don't have location info.")
    else:
        print("FAQ tool couldn't find a match.")
        return {"task_result": "I couldn't find a direct answer in the FAQ.", "error_message": "FAQ Not Found"}
    print(f"FAQ Result: {result}\n")
    return {"task_result": result, "error_message": None}

def human_handoff_node(state: AppState) -> dict:
    # (No changes needed from previous version)
    print("--- Executing Human Handoff ---")
    result = "Connecting you to a human agent..."
    print(result + "\n")
    return {"task_result": result, "next_action": END, "error_message": None}

# -----------------------------------------------------------------------------
# Level 2: Departmental Supervisors (Updated)

def l2_auth_supervisor(state: AppState) -> dict:
    """Supervisor for authentication tasks (login, signup, reset, logout)."""
    print("--- L2 Auth Supervisor ---")
    task = state.get("current_task", "").lower()

    # Check for logout first as it's a distinct action
    if "logout" in task or "log out" in task or "sign out" in task:
         next_action = "L3_LogoutToolNode"
    elif "login" in task or "sign in" in task:
        next_action = "L3_LoginToolNode"
    elif "signup" in task or "register" in task or "new account" in task:
        next_action = "L3_SignupToolNode"
    elif "password" in task or "reset" in task or "forgot" in task:
        next_action = "L3_PasswordResetToolNode"
    else:
        # If an auth-related task was routed here but isn't recognized
        print(f"L2 Auth Warning: Unknown task '{task}'. Assuming login required or routing to Support.")
        # If user is logged in, maybe it's a misunderstanding, route to support.
        # If user is NOT logged in, it's safer to assume they might need to log in.
        if state.get("user_info"):
             return {"next_action": "L2_SupportSupervisor", "error_message": "Unknown Auth Task"}
        else:
             # Default to login if task is unclear and user isn't logged in.
             # Or, could route to support with a clearer message. Let's route to support.
             print("Task unclear, routing to support.")
             return {"next_action": "L2_SupportSupervisor", "error_message": "Unknown Auth Task"}

    print(f"Routing to: {next_action}")
    return {"next_action": next_action, "error_message": None}


def l2_account_supervisor(state: AppState) -> dict:
    """Supervisor for account management tasks."""
    print("--- L2 Account Supervisor ---")
    task = state.get("current_task", "").lower()

    if not state.get("user_info"):
        print("Authentication required for account actions. Routing to Auth.")
        original_task = state.get("current_task", "your request")
        return {
            "next_action": "L2_AuthSupervisor",
            "current_task": f"login (required for: {original_task})",
            "error_message": "Authentication Required"
        }

    # --- Task Routing within Account ---
    if "balance" in task:
        next_action = "L3_CheckBalanceToolNode"
    elif "history" in task or "transaction" in task or "statement" in task:
        next_action = "L3_GetHistoryToolNode"
    # *** UPDATED: Route to specific name update tools ***
    elif ("update" in task or "change" in task) and "account holder name" in task:
         next_action = "L3_AccountHolderNameUpdateToolNode"
    elif ("update" in task or "change" in task) and ("login name" in task or "display name" in task):
         next_action = "L3_LoginNameUpdateToolNode"
    # Handle ambiguous "update name" - maybe ask for clarification or default?
    # Let's default to account holder name change if ambiguous, as it's often more critical.
    elif ("update" in task or "change" in task or "correct" in task) and "name" in task:
         print("Ambiguous 'update name' request. Assuming Account Holder Name.")
         next_action = "L3_AccountHolderNameUpdateToolNode"
    # *** UPDATED: Display account details with both names ***
    elif "account number" in task or "account id" in task or "account type" in task or "account details" in task:
        user_info = state["user_info"]
        details = (
            # Display both names clearly
            f"Login/Display Name: {user_info.get('name', 'N/A')}\n"
            f"Account Holder Name: {user_info.get('account_holder_name', 'N/A')}\n"
            f"Account Number: {user_info.get('account_number', 'N/A')}\n"
            f"Account Type: {user_info.get('account_type', 'N/A')}\n"
            f"Account ID: {user_info.get('account_id', 'N/A')}"
        )
        print("--- L2 Account Supervisor: Displaying Details ---")
        print(details + "\n")
        return {"task_result": details, "next_action": END, "error_message": None}
    else:
        print(f"L2 Account Warning: Unknown task '{task}'. Routing to Support.")
        return {"next_action": "L2_SupportSupervisor", "error_message": "Unknown Account Task"}

    print(f"Routing to: {next_action}")
    return {"next_action": next_action, "error_message": None}


def l2_support_supervisor(state: AppState) -> dict:
    # (No significant changes needed, acts as fallback and FAQ handler)
    print("--- L2 Support Supervisor ---")
    task = state.get("current_task", "").lower()
    last_user_message = ""
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content.lower()
            break

    if state.get("error_message") == "FAQ Not Found":
        print("Handling 'FAQ Not Found' error, suggesting handoff.")
        next_action = "L3_HumanHandoffNode"
    elif any(kw in task or kw in last_user_message for kw in ["hour", "open", "contact", "phone", "call", "location", "address", "branch"]):
         next_action = "L3_FAQToolNode"
    elif any(kw in task or kw in last_user_message for kw in ["help", "support", "issue", "human", "agent", "problem", "talk to someone"]):
         next_action = "L3_HumanHandoffNode"
    else:
         # Handle errors specifically routed here (like Unknown Auth/Account Task)
         if state.get("error_message") in ["Unknown Auth Task", "Unknown Account Task", "L1 Routing Failed"]:
             print(f"Handling error '{state.get('error_message')}'. Suggesting handoff.")
             next_action = "L3_HumanHandoffNode"
         else:
             print("L2 Support: Task unclear or generic, suggesting handoff.")
             next_action = "L3_HumanHandoffNode"

    print(f"Routing to: {next_action}")
    # Clear error if we explicitly handled it here (like FAQ Not Found)
    # Keep other errors (like Unknown Task) for potential logging/analysis if needed
    error_msg = state.get("error_message") if state.get("error_message") != "FAQ Not Found" else None
    return {"next_action": next_action, "error_message": error_msg}


# -----------------------------------------------------------------------------
# Level 1: Main Orchestrator / Supervisor - UPDATED PROMPT

L1_SUPERVISOR_SYSTEM_PROMPT = """You are the main orchestrator for a multi-agent banking assistant.
Your goal is to understand the user's request and route it to the correct department supervisor.
The available departments and their primary functions are:
1.  **Auth**: Handles **login**, **signup**, **password reset/forgot password**, and **logout** requests.
2.  **Account**: Handles requests about checking **balance**, **transaction history**, viewing **account details** (number, ID, type), updating the official **account holder name**, and updating the **login/display name**. Requires the user to be logged in.
3.  **Support**: Handles general questions (**FAQs** like hours, contact), requests for **human help**, or issues not covered by other departments (fallback).

Based on the user's latest message and the conversation history, determine the primary intent.
Consider if the user is currently authenticated (logged in). Some Account department actions require authentication.

Respond *only* with the name of the department supervisor to route to:
- L2_AuthSupervisor
- L2_AccountSupervisor
- L2_SupportSupervisor

**Routing Rules:**
- If the user asks to **logout/log out/sign out**, route to `L2_AuthSupervisor`.
- If the user needs to perform an **Account** action (balance, history, update names, view details) but is **Not Logged In**, route to `L2_AuthSupervisor` first to handle login.
- If the user asks to update/change their **account holder name**, route to `L2_AccountSupervisor` (if logged in).
- If the user asks to update/change their **login name** or **display name**, route to `L2_AccountSupervisor` (if logged in).
- If the request is ambiguous about which name to update, route to `L2_AccountSupervisor`.
- For general help, FAQs, or unclear requests, route to `L2_SupportSupervisor`.

**Examples:**
User: "Log me out" -> Respond: L2_AuthSupervisor
User: "What's my balance?" (Logged In) -> Respond: L2_AccountSupervisor
User: "What's my balance?" (Not Logged In) -> Respond: L2_AuthSupervisor
User: "I want to change the name on my account" (Logged In) -> Respond: L2_AccountSupervisor
User: "Update my login name" (Logged In) -> Respond: L2_AccountSupervisor
User: "I want to change my name" (Not Logged In) -> Respond: L2_AuthSupervisor
User: "I forgot my password" -> Respond: L2_AuthSupervisor
User: "What are your hours?" -> Respond: L2_SupportSupervisor
User: "Help me with something" -> Respond: L2_SupportSupervisor
"""

# l1_main_supervisor function uses the updated prompt but logic remains the same
def l1_main_supervisor(state: AppState) -> dict:
    print("\n--- L1 Main Supervisor ---")
    messages = state['messages']
    user_info = state.get('user_info')
    auth_status = "Logged In" if user_info else "Not Logged In"
    print(f"Current Auth Status: {auth_status}")

    prompt_messages = [SystemMessage(content=L1_SUPERVISOR_SYSTEM_PROMPT)]
    prompt_messages.append(SystemMessage(content=f"Current Authentication Status: {auth_status}"))
    history_limit = 5
    prompt_messages.extend(messages[-(history_limit):])

    try:
        response = llm.invoke(prompt_messages)
        llm_decision = response.content.strip()
        print(f"L1 LLM Decision: {llm_decision}")

        valid_routes = ["L2_AuthSupervisor", "L2_AccountSupervisor", "L2_SupportSupervisor"]
        if llm_decision in valid_routes:
            next_action = llm_decision
            last_user_message = ""
            for msg in reversed(messages):
                 if isinstance(msg, HumanMessage):
                     last_user_message = msg.content
                     break
            current_task = last_user_message
            print(f"Routing to: {next_action} with task: '{current_task}'")
            return {"next_action": next_action, "current_task": current_task, "task_result": None, "error_message": None}
        else:
            print(f"L1 Warning: LLM produced invalid route '{llm_decision}'. Defaulting to Support.")
            last_user_message = ""
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    last_user_message = msg.content
                    break
            return {"next_action": "L2_SupportSupervisor", "current_task": last_user_message or "Unknown intent", "error_message": "L1 Routing Failed"}

    except Exception as e:
        print(f"L1 Error: Exception during LLM call: {e}")
        last_user_message = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break
        return {"next_action": "L2_SupportSupervisor", "current_task": last_user_message or "System error", "error_message": f"L1 LLM Error: {e}"}


# -----------------------------------------------------------------------------
# Graph Definition - UPDATED with new/renamed nodes and routing

# Conditional routing functions remain the same logic
def route_l1_decision(state: AppState) -> str:
    next_node = state.get("next_action")
    print(f"[Router] L1 Decision: Route to {next_node}")
    if next_node in ["L2_AuthSupervisor", "L2_AccountSupervisor", "L2_SupportSupervisor"]:
        return next_node
    else:
        print(f"[Router] L1 Warning/Error: Invalid next_action '{next_node}'. Defaulting to Support.")
        # Ensure error message is set for L2 Support
        state["error_message"] = state.get("error_message") or f"Invalid L1 Routing Decision: {next_node}"
        return "L2_SupportSupervisor"

def route_l2_decision(state: AppState) -> str:
    next_node = state.get("next_action")
    print(f"[Router] L2 Decision: Route to {next_node}")
    # *** UPDATED List of ALL valid L3 tool nodes ***
    valid_l3_nodes = [
        "L3_LoginToolNode", "L3_SignupToolNode", "L3_CheckBalanceToolNode",
        "L3_GetHistoryToolNode", "L3_FAQToolNode", "L3_HumanHandoffNode",
        "L3_PasswordResetToolNode",
        "L3_LogoutToolNode",                  # New
        "L3_LoginNameUpdateToolNode",         # Renamed
        "L3_AccountHolderNameUpdateToolNode"  # New
    ]
    valid_l2_reroutes = ["L2_AuthSupervisor", "L2_SupportSupervisor"]
    if next_node in valid_l3_nodes:
        return next_node
    elif next_node in valid_l2_reroutes:
        return next_node
    elif next_node == END:
         return END
    else:
        print(f"[Router] L2 Warning: Invalid next_action '{next_node}'. Defaulting to Support.")
        state["error_message"] = state.get("error_message") or f"Invalid L2 Routing Decision: {next_node}"
        return "L2_SupportSupervisor"

def route_after_tool(state: AppState) -> str:
    print(f"[Router] After Tool Execution...")
    error_message = state.get("error_message")
    next_action_forced_by_tool = state.get("next_action")

    if next_action_forced_by_tool == END:
        print("Routing to END (forced by tool).")
        return END

    if error_message:
        print(f"Error detected after tool: {error_message}.")
        # Specific error routing
        if error_message in ["Authentication Required", "Authentication Failed", "Account Data Mismatch"]:
             print("Authentication error detected. Routing to L2 Auth Supervisor.")
             # Clear the error message as L2 Auth will handle the flow (or login tool failed)
             # Keep current_task as it might be relevant (e.g., "login required for X")
             state["error_message"] = None # Clear error before re-routing to avoid loops
             # Ensure user_info is cleared if mismatch/failed
             if error_message != "Authentication Required":
                 state["user_info"] = None
             return "L2_AuthSupervisor" # Try auth flow again
        elif error_message == "FAQ Not Found":
             print("FAQ not found. Routing to L2 Support Supervisor.")
             # Keep error message for L2 Support
             return "L2_SupportSupervisor"
        # Add specific handling for other known, recoverable errors if needed
        # elif error_message == "Invalid New Name": ... maybe route back to L2 Account? Or END?
        else:
             print("Ending turn due to unhandled tool error.")
             return END # End the turn, main loop will display error
    else:
        # Successful tool execution ends the current turn.
        print("Tool executed successfully. Ending current turn.")
        return END


# Build the graph
builder = StateGraph(AppState)

# Add Nodes (including NEW and RENAMED ones)
builder.add_node("L1_Supervisor", l1_main_supervisor)
builder.add_node("L2_AuthSupervisor", l2_auth_supervisor)
builder.add_node("L2_AccountSupervisor", l2_account_supervisor)
builder.add_node("L2_SupportSupervisor", l2_support_supervisor)

# Auth Tools
builder.add_node("L3_LoginToolNode", login_tool_node)
builder.add_node("L3_SignupToolNode", signup_tool_node)
builder.add_node("L3_PasswordResetToolNode", password_reset_tool_node)
builder.add_node("L3_LogoutToolNode", logout_tool_node) # New

# Account Tools
builder.add_node("L3_CheckBalanceToolNode", check_balance_tool_node)
builder.add_node("L3_GetHistoryToolNode", get_history_tool_node)
builder.add_node("L3_LoginNameUpdateToolNode", login_name_update_tool_node) # Renamed
builder.add_node("L3_AccountHolderNameUpdateToolNode", account_holder_name_update_tool_node) # New

# Support Tools
builder.add_node("L3_FAQToolNode", faq_tool_node)
builder.add_node("L3_HumanHandoffNode", human_handoff_node)


# Define Edges

builder.add_edge(START, "L1_Supervisor")

# L1 to L2 routing
builder.add_conditional_edges(
    "L1_Supervisor",
    route_l1_decision,
    {
        "L2_AuthSupervisor": "L2_AuthSupervisor",
        "L2_AccountSupervisor": "L2_AccountSupervisor",
        "L2_SupportSupervisor": "L2_SupportSupervisor",
    }
)

# *** UPDATED L3_TOOLS list ***
L3_TOOLS = [
    "L3_LoginToolNode", "L3_SignupToolNode", "L3_PasswordResetToolNode", "L3_LogoutToolNode",
    "L3_CheckBalanceToolNode", "L3_GetHistoryToolNode", "L3_LoginNameUpdateToolNode", "L3_AccountHolderNameUpdateToolNode",
    "L3_FAQToolNode", "L3_HumanHandoffNode"
]
L2_SUPERVISORS = ["L2_AuthSupervisor", "L2_AccountSupervisor", "L2_SupportSupervisor"]

# L2 to L3/L2/END routing
l2_conditional_map = {node: node for node in L3_TOOLS} # Map tool names to themselves
l2_conditional_map["L2_AuthSupervisor"] = "L2_AuthSupervisor" # Re-route to Auth
l2_conditional_map["L2_SupportSupervisor"] = "L2_SupportSupervisor" # Re-route to Support
l2_conditional_map[END] = END # Allow L2 to end directly

for supervisor_node in L2_SUPERVISORS:
    builder.add_conditional_edges(
        supervisor_node,
        route_l2_decision,
        l2_conditional_map
    )

# Routing after L3 tools execute
after_tool_map = {
    "L2_AuthSupervisor": "L2_AuthSupervisor", # For specific error re-routing
    "L2_SupportSupervisor": "L2_SupportSupervisor", # For specific error re-routing
    END: END                            # Default success/handled error path
}
for tool_node in L3_TOOLS:
     builder.add_conditional_edges(
         tool_node,
         route_after_tool,
         after_tool_map
     )

# Compile the graph
try:
    graph = builder.compile()
    print("\nGraph compiled successfully!")
except Exception as e:
    print(f"\nFatal Error: Graph compilation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Visualize (Optional)
try:
    output_filename = "banking_agent_graph_v3.png"
    graph.get_graph().draw_mermaid_png(output_file_path=output_filename)
    print(f"Graph visualization saved to {output_filename}")
except ImportError:
    print("Install pygraphviz and graphviz to visualize the graph: pip install pygraphviz")
except Exception as e:
    print(f"Warning: Could not generate graph visualization: {e}")


# -----------------------------------------------------------------------------
# Main conversation loop (Using direct list update for messages)

def main():
    print("\n=== Welcome to the Multi-Level Banking Assistant (v3) ===")
    print("You can ask about balance, history, FAQs, login, signup, password reset, logout, name updates.")
    print("Type 'quit' or 'exit' to end the conversation.")

    current_state: AppState = {
        "messages": [],
        "user_info": None,
        "current_task": None,
        "task_result": None,
        "next_action": None,
        "error_message": None,
    }

    while True:
        # Display login status clearly
        login_display_name = current_state.get("user_info", {}).get("name") if current_state.get("user_info") else None
        auth_display = f"(Logged in as: {login_display_name})" if login_display_name else "(Not Logged In)"
        user_input = input(f"\nYou {auth_display}: ")

        if 'quit' in user_input.lower() or 'exit' in user_input.lower():
            print("Banking Assistant: Goodbye!")
            break

        # --- Prepare state for the new turn ---
        # 1. Add user message directly to the list
        current_messages = current_state.get('messages', [])
        if not isinstance(current_messages, list):
             current_messages = list(current_messages)
        current_messages.append(HumanMessage(content=user_input))
        current_state['messages'] = current_messages

        # 2. Clear results/errors from the *previous* turn
        current_state['task_result'] = None
        current_state['error_message'] = None
        current_state['next_action'] = None # MUST be cleared before graph invocation

        print("\nAssistant Processing...")
        try:
            # Stream updates and update the state progressively
            for event in graph.stream(current_state, {"recursion_limit": 25}):
                # The event dictionary's key is the node name
                node_name = list(event.keys())[0]
                # The value is the dictionary returned by the node
                node_output = event[node_name]
                print(f"--- Event: Node '{node_name}' Output: {node_output} ---")
                # Merge the partial update from the node into the central state
                current_state.update(node_output)

            # --- After stream finishes for this input ---
            final_task_result = current_state.get("task_result")
            final_error_message = current_state.get("error_message")

            # Display final result or error for the turn
            if final_task_result:
                 print(f"\nAssistant: {final_task_result}")
                 # Optional: Add AI response to history if desired
                 # current_state['messages'].append(AIMessage(content=final_task_result))
            elif final_error_message:
                 print(f"\nAssistant: Error: {final_error_message}. Please try again or ask for help.")
                 # Optional: Add AI error response to history
                 # current_state['messages'].append(AIMessage(content=f"Error: {final_error_message}"))
            else:
                 # This case should be less common now as END is usually forced by route_after_tool
                 # It might happen if a supervisor routes directly to END without a task_result
                 if current_state.get("next_action") == END:
                     # A node forced END without a specific message (e.g., account details display)
                     print("\nAssistant: Request completed.") # Generic completion message
                 else:
                     # Graph ended unexpectedly without result/error/END signal
                     print("\nAssistant: How else can I assist you today?")


        except Exception as e:
            print(f"\n--- Critical Error during graph execution ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            import traceback
            traceback.print_exc()
            print("\nAssistant: I've encountered a critical system error. Please restart the conversation or try again later.")
            # Consider breaking the loop on critical errors
            break

if __name__ == "__main__":
    main()