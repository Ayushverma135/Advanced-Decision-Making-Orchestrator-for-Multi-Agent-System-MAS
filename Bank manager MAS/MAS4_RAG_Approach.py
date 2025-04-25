# --- Imports ---
import random
import string
import uuid
from typing import TypedDict, List, Optional, Sequence, Annotated
import torch # Still useful for device detection
import traceback # For error printing

# --- Imports for Langchain Embedding approach ---
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("langchain-huggingface library loaded successfully.")
except ImportError:
    print("Fatal Error: langchain-huggingface library not found.")
    print("Please install it: pip install langchain-huggingface")
    exit(1)

# --- Imports for Similarity Calculation ---
try:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    print("scikit-learn and numpy loaded for similarity calculation.")
except ImportError:
    print("Fatal Error: scikit-learn or numpy not found.")
    print("Please install them: pip install scikit-learn numpy")
    exit(1)

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# -----------------------------------------------------------------------------
# Helper functions, DBs, State Definition (unchanged)
# -----------------------------------------------------------------------------
# Helper functions (unchanged)
def generate_account_number():
    return ''.join(random.choices(string.digits, k=10))

def generate_account_id():
    return str(uuid.uuid4())

# -----------------------------------------------------------------------------
# Local in-memory database (unchanged)
local_db = {
    "ayush@gmail.com": {
        "name": "ayush135",
        "account_holder_name": "Ayush Sharma",
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
# State Definition (unchanged)
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
    current_task: Optional[str]
    task_result: Optional[str]
    next_action: Optional[str]
    error_message: Optional[str]

# -----------------------------------------------------------------------------
# Level 3: Specialist Tools / Nodes (unchanged)
# --- Authentication Tools ---
def login_tool_node(state: AppState) -> dict:
    """Node to handle the login process and load user info into state."""
    print("--- Executing Login Tool ---")
    email = input("Enter your email: ")
    password = input("Enter your password: ")
    user_data = local_db.get(email)
    if user_data and user_data["password"] == password:
        user_info: UserInfo = {
            "email": email,
            "name": user_data.get("name", "User"),
            "account_holder_name": user_data.get("account_holder_name", "N/A"),
            "account_number": user_data.get("account_number", "N/A"),
            "account_id": user_data.get("account_id", "N/A"),
            "account_type": user_data.get("account_type", "N/A")
        }
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
    account_holder_name = input("Enter the full name for the account holder: ")
    password = input("Enter your password: ")

    if not login_name or not account_holder_name or not password:
         result = "Error: All fields (email, names, password) are required."
         print(result + "\n")
         return {"task_result": result, "error_message": "Missing Signup Field(s)"}

    account_number = generate_account_number()
    account_id = generate_account_id()
    account_type = "Savings" # Default

    local_db[email] = {
        "name": login_name.strip(),
        "account_holder_name": account_holder_name.strip(),
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

def logout_tool_node(state: AppState) -> dict:
    """Node to handle user logout."""
    print("--- Executing Logout Tool ---")
    if not state.get("user_info"):
        result = "You are not currently logged in."
        print(result + "\n")
        return {"task_result": result, "error_message": None}
    else:
        logged_in_name = state["user_info"].get("name", "User")
        result = f"Logging out {logged_in_name}. You have been logged out successfully."
        print(result + "\n")
        return {"user_info": None, "task_result": result, "error_message": None}

# --- Account Management Tools --- (unchanged)
def check_balance_tool_node(state: AppState) -> dict:
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
        local_db[email]["name"] = new_login_name
        updated_user_info = user_info.copy()
        updated_user_info["name"] = new_login_name

        result = f"Your login/display name has been updated to '{new_login_name}'."
        print(result + "\n")
        return {"task_result": result, "user_info": updated_user_info, "error_message": None}
    else:
        result = "Error: Could not find your account details to update login name."
        print(result + "\n")
        return {"task_result": result, "error_message": "Account Data Mismatch", "user_info": None}

def account_holder_name_update_tool_node(state: AppState) -> dict:
    """Node to update the official account holder name."""
    print("--- Executing Account Holder Name Update Tool ---")
    if not state.get("user_info"):
        result = "Error: You must be logged in to update the account holder name."
        print(result + "\n")
        return {"task_result": result, "error_message": "Authentication Required"}

    user_info = state["user_info"]
    email = user_info["email"]
    current_holder_name = user_info.get("account_holder_name", "N/A")
    new_holder_name = input(f"The current account holder name is '{current_holder_name}'. Enter the new full name for the account holder: ")

    if not new_holder_name or new_holder_name.strip() == "":
        result = "Error: New account holder name cannot be empty."
        print(result + "\n")
        return {"task_result": result, "error_message": "Invalid New Account Holder Name"}

    new_holder_name = new_holder_name.strip()

    if email in local_db:
        local_db[email]["account_holder_name"] = new_holder_name
        updated_user_info = user_info.copy()
        updated_user_info["account_holder_name"] = new_holder_name

        result = f"The account holder name has been updated to '{new_holder_name}'."
        print(result + "\n")
        return {"task_result": result, "user_info": updated_user_info, "error_message": None}
    else:
        result = "Error: Could not find your account details to update account holder name."
        print(result + "\n")
        return {"task_result": result, "error_message": "Account Data Mismatch", "user_info": None}

# --- Support Tools --- (unchanged)
def faq_tool_node(state: AppState) -> dict:
    print("--- Executing FAQ Tool ---")
    last_user_message = ""
    # Check the 'current_task' passed down for keywords first
    task = state.get("current_task", "").lower()
    # Fallback to last message if task isn't informative (e.g., if L1 passed a generic task)
    if not any(kw in task for kw in ["hour", "open", "contact", "phone", "location", "address"]):
         for msg in reversed(state['messages']):
             if isinstance(msg, HumanMessage):
                 last_user_message = msg.content.lower()
                 break
    else:
        last_user_message = task # Use the task if it seems relevant

    if "hour" in last_user_message or "open" in last_user_message:
        result = faq_db.get("hours", "Sorry, I don't have info on hours.")
    elif "contact" in last_user_message or "phone" in last_user_message:
         result = faq_db.get("contact", "Sorry, I don't have contact info.")
    elif "location" in last_user_message or "address" in last_user_message:
         result = faq_db.get("locations", "Sorry, I don't have location info.")
    else:
        print("FAQ tool couldn't find a match based on task/message.")
        # This path might be less likely if L2 Support RAG directs here accurately
        return {"task_result": "I couldn't find a direct answer in the FAQ.", "error_message": "FAQ Not Found"}

    print(f"FAQ Result: {result}\n")
    return {"task_result": result, "error_message": None}


def human_handoff_node(state: AppState) -> dict:
    print("--- Executing Human Handoff ---")
    result = "Connecting you to a human agent..."
    print(result + "\n")
    return {"task_result": result, "next_action": END, "error_message": None}

# -----------------------------------------------------------------------------
# --- RAG Setup (L1 and L2) ---

# 1. Initialize Embedding Model (once)
hf_model_name = "sentence-transformers/all-MiniLM-L6-v2"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = None
try:
    print(f"Using device: {device}")
    print(f"Loading HuggingFaceEmbeddings model: {hf_model_name}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=hf_model_name,
        model_kwargs={'device': device},
    )
    _ = embedding_model.embed_query("test initialization") # Test query
    print("HuggingFaceEmbeddings model loaded successfully.")
except Exception as e:
    print(f"\n--- Fatal Error Initializing Embedding Model ---")
    print(f"Failed to load model: {hf_model_name}")
    print(f"Error: {e}")
    traceback.print_exc()
    exit(1)

# --- L1 Supervisor RAG Data ---
l1_route_definitions = [
    {
        "name": "L2_AuthSupervisor",
        "description": (
            "Handles user access, login, and security credentials. "
            "Keywords: log in, sign in, authenticate, access my account, enter credentials. "
            "Keywords: log out, sign out, exit session, disconnect. "
            "Keywords: register, sign up, create account, new user, open account, enroll. "
            "Keywords: forgot password, reset password, change password, update password, security question, cannot login, locked out."
            "Focus: Gaining or losing access to the application/service, managing the password."
        )
    },
    {
        "name": "L2_AccountSupervisor",
        "description": (
            "Provides information about or modifies details of a specific, existing bank account (requires user to be logged in). "
            "Keywords: check balance, view balance, how much money, available funds. "
            "Keywords: transaction history, view transactions, recent activity, statement, past payments, spending, deposits, withdrawals. "
            "Keywords: account details, view account number, see account type, account ID. "
            "Keywords: update name, change name, correct name (distinguish carefully below). "
            "Keywords: update *account holder* name, change *legal* name, correct *official* name. "
            "Keywords: update *login* name, change *display* name, username, nickname, profile name. "
            "Focus: Information *within* an already authenticated account or changes to account *metadata* (like names associated with it)."
        )
    },
    {
        "name": "L2_SupportSupervisor",
        "description": (
            "Handles general bank information (FAQs), requests for human assistance, and fallback for unclear queries. "
            "Keywords: hours, opening times, when are you open. "
            "Keywords: locations, address, where are branches, find branch. "
            "Keywords: contact, phone number, call us, email address, customer service number. "
            "Keywords: help, support, assistance, need help, problem, issue, error, complaint, feedback. "
            "Keywords: talk to someone, human agent, representative, speak to person. "
            "Focus: General bank operations, getting help with the service, or when the intent isn't clearly authentication or account data management."
        )
    }
]
l1_route_names = [r["name"] for r in l1_route_definitions]
l1_route_embeddings_np = np.array(embedding_model.embed_documents([r["description"] for r in l1_route_definitions]))
print(f"L1 route embeddings computed (shape: {l1_route_embeddings_np.shape}).")

# --- L2 Auth Supervisor RAG Data ---
l2_auth_tool_definitions = [
    {
        "name": "L3_LoginToolNode",
        "description": (
            "Handles the process of a user entering their credentials to gain access to their existing account. "
            "Keywords: log in, sign in, logon, authenticate, access my account, get into my account, enter credentials, user login, member access. "
            "User might say: 'I want to log in', 'Let me sign in', 'Need to access my banking', 'How do I get into my account?'"
            "Focus: Gaining entry with existing credentials."
        )
    },
    {
        "name": "L3_SignupToolNode",
        "description": (
            "Handles the process for a new user creating an account for the first time. "
            "Keywords: register, sign up, create account, new account, open account, enroll, join, become a member, new user registration. "
            "User might say: 'I need to sign up', 'How do I register?', 'Create a new profile', 'I don't have an account yet', 'Get started'."
            "Focus: Creating brand new user access."
        )
    },
    {
        "name": "L3_PasswordResetToolNode",
        "description": (
            "Handles situations where a user cannot log in due to a forgotten password, or wants to change their existing password. "
            "Keywords: forgot password, reset password, change password, update password, new password, password recovery, password help, password issue, can't log in, invalid password, locked out, security credentials problem, recover account access. "
            "User might say: 'I forgot my password', 'Need to reset my login password', 'Help with password', 'My password isn't working', 'Update my security details', 'Change existing password'."
            "Focus: Resolving password-related access issues or changing the security credential."
        )
    },
    {
        "name": "L3_LogoutToolNode",
        "description": (
            "Handles the process of ending the user's current authenticated session. "
            "Keywords: log out, sign out, exit, leave, end session, disconnect, close session, secure exit. "
            "User might say: 'Log me out', 'Sign out of my account', 'I want to leave', 'End my banking session', 'Disconnect me'."
            "Focus: Terminating the current active session."
        )
    }
]

l2_auth_tool_names = [t["name"] for t in l2_auth_tool_definitions]
l2_auth_tool_embeddings_np = np.array(embedding_model.embed_documents([t["description"] for t in l2_auth_tool_definitions]))
print(f"L2 Auth tool embeddings computed (shape: {l2_auth_tool_embeddings_np.shape}).")

# --- L2 Account Supervisor RAG Data ---
# Special constant for the direct display action handled within L2
DISPLAY_DETAILS_ACTION = "L2_DISPLAY_DETAILS"
l2_account_tool_definitions = [
    {
        "name": "L3_CheckBalanceToolNode",
        "description": (
            "Handles requests to check the current available funds or monetary balance in the user's account. "
            "Keywords: check balance, view balance, current balance, available funds, account funds, how much money, amount available, funds status, remaining money. "
            "User might say: 'What's my balance?', 'How much money do I have?', 'Show my available funds', 'Tell me the balance', 'Check my current amount', 'Do I have enough money for X?'"
            "Focus: Retrieving the current monetary value available in the account."
        )
    },
    {
        "name": "L3_GetHistoryToolNode",
        "description": (
            "Handles requests to view a list or record of past financial activities (deposits, withdrawals, payments, transfers) on the account. "
            "Keywords: transaction history, view transactions, recent activity, statement, past payments, spending history, deposit list, withdrawal record, activity log, account statement, transaction list, payment history, money movements. "
            "User might say: 'Show my recent transactions', 'I need my statement for last month', 'Where did my money go?', 'List recent payments', 'View my activity log', 'See my past deposits', 'Check withdrawal history'."
            "Focus: Listing past financial events and money movements."
        )
    },
    {
        "name": "L3_AccountHolderNameUpdateToolNode",
        "description": (
            "Handles requests to change the **official legal name** associated with the account holder (e.g., due to marriage, divorce, legal name change). This is the formal name on the account. "
            "Keywords: update account holder name, change legal name, correct official name, fix name spelling, update primary name, name change documentation, maiden name update, new last name, formal name correction. "
            "User might say: 'I need to change my name on the account after getting married', 'Update my last name legally', 'My legal name changed, update my bank account', 'Fix the spelling of my official name on the account', 'Submit name change documents'."
            "Focus: Modifying the **official/legal** identity tied to the account. Requires verification."
        )
    },
    {
        "name": "L3_LoginNameUpdateToolNode",
        "description": (
            "Handles requests to change the **login username or display name** used within the application or online banking portal (this is *not* the official legal account holder name). "
            "Keywords: update login name, change display name, username, screen name, nickname, profile name, alias, login ID, user ID change, online banking name. "
            "User might say: 'Change my username for logging in', 'I want a different display name on the app', 'Update my profile nickname', 'How to change my login ID?', 'Edit my screen name for online banking'."
            "Focus: Modifying the **informal/display** identity used for login or personalization within the service."
        )
    },
    {
        "name": DISPLAY_DETAILS_ACTION, # Special case handled directly in L2 Supervisor
        "description": (
            "Handles requests to view static information and identifying details about the account itself, such as numbers and type. "
            "Keywords: account details, view account number, see account type, account ID, account summary, routing number, IBAN, sort code, account info, basic account information. "
            "User might say: 'What's my account number?', 'Show my full account details', 'Tell me my account type (Savings/Checking)', 'I need my routing number', 'View account summary', 'Find my account ID'."
            "Focus: Displaying fixed identifiers and metadata of the specific bank account."
        )
    }
]
l2_account_tool_names = [t["name"] for t in l2_account_tool_definitions]
l2_account_tool_embeddings_np = np.array(embedding_model.embed_documents([t["description"] for t in l2_account_tool_definitions]))
print(f"L2 Account tool embeddings computed (shape: {l2_account_tool_embeddings_np.shape}).")

# --- L2 Support Supervisor RAG Data ---
l2_support_tool_definitions = [
    {
        "name": "L3_FAQToolNode",
        "description": (
            "Handles requests for standard, publicly available information about the bank's operations, contact methods, locations, or general product features (FAQs). Does not handle specific account data. "
            "Keywords: hours, opening times, business hours, holiday hours, when are you open/closed. "
            "Keywords: contact info, phone number, call us, customer service number, email address, mailing address, reach out. "
            "Keywords: location, address, branch finder, find a branch, nearby branch, ATM location, where are you located. "
            "Keywords: fees, charges, overdraft fee, wire transfer fee, monthly fee, service costs. "
            "Keywords: interest rates (general inquiry), savings rate, loan rate information. "
            "Keywords: website help, app help, navigation assistance (general). "
            "User might say: 'What are your Saturday hours?', 'How can I call customer support?', 'Find the branch near me', 'What's the fee for an international wire?', 'Where can I find ATM locations?', 'Are you open on Memorial Day?', 'Tell me the savings account interest rate.'"
            "Focus: Providing pre-defined, factual answers to common questions about the bank itself."
        )
    },
    {
        "name": "L3_HumanHandoffNode",
        "description": (
            "Handles situations where the user explicitly requests human help, expresses significant frustration, reports serious issues (like fraud), or has a query too complex or ambiguous for the automated tools. This is the fallback route. "
            "Keywords: help, support, assistance, need help, confused, frustrated, stuck, this isn't working. "
            "Keywords: agent, human, representative, person, talk to someone, speak to manager, operator, customer service representative. "
            "Keywords: issue, problem, error, complaint, feedback, report issue, technical difficulty. "
            "Keywords: security concern, fraud alert, unauthorized transaction, report stolen card, account security. "
            "Keywords: complex situation, specific account problem, need advice, personalized help, escalate issue, bypass bot, override. "
            "Keywords: unclear request, ambiguous query, I don't understand, something else. "
            "User might say: 'I need to speak to a human now', 'Connect me with an agent', 'I'm having trouble and need help', 'This bot isn't understanding me', 'Report a fraudulent charge', 'My problem is complicated', 'Can I get advice on loans?', 'Just transfer me to support', 'Help with something not listed'."
            "Focus: Escalating the interaction to a human employee for resolution due to user request, complexity, ambiguity, errors, or sensitive issues."
        )
    }
]
l2_support_tool_names = [t["name"] for t in l2_support_tool_definitions]
l2_support_tool_embeddings_np = np.array(embedding_model.embed_documents([t["description"] for t in l2_support_tool_definitions]))
print(f"L2 Support tool embeddings computed (shape: {l2_support_tool_embeddings_np.shape}).")

# --- RAG Utility Function ---
# Corrected syntax using tuple[...]
def find_best_match(query: str, names: List[str], embeddings: np.ndarray, confidence_threshold=0.3) -> tuple[Optional[str], float]:
    """Embeds query and finds the best match in pre-computed embeddings."""
    if not query or not query.strip():
        print("[RAG Util] Warning: Empty query provided.")
        return None, 0.0 # Returning the actual values is fine

    try:
        query_embedding_list = embedding_model.embed_query(query)
        query_embedding_np = np.array(query_embedding_list).reshape(1, -1)

        cosine_scores = cosine_similarity(query_embedding_np, embeddings)[0]
        best_match_idx = np.argmax(cosine_scores)
        best_score = cosine_scores[best_match_idx]

        # Debugging: Print scores
        scores_dict = {name: score.item() for name, score in zip(names, cosine_scores)}
        print(f"[RAG Util] Similarities: { {k: f'{v:.4f}' for k, v in scores_dict.items()} } ")

        if best_score >= confidence_threshold:
            best_match_name = names[best_match_idx]
            print(f"[RAG Util] Best Match: {best_match_name} (Score: {best_score:.4f})")
            return best_match_name, float(best_score) # Ensure float return
        else:
            print(f"[RAG Util] Low Confidence: Best score {best_score:.4f} below threshold {confidence_threshold}. No match.")
            return None, float(best_score) # Ensure float return

    except Exception as e:
        print(f"[RAG Util] Error during embedding/similarity: {e}")
        traceback.print_exc()
        return None, 0.0 # Return default values matching the type hint

# -----------------------------------------------------------------------------
# --- Level 1 Supervisor (RAG Based - using utility function) ---
def l1_main_supervisor_rag(state: AppState) -> dict:
    """Routes user requests to L2 supervisors using RAG."""
    print("\n--- L1 Main Supervisor (Langchain RAG Based) ---")
    is_logged_in = bool(state.get('user_info'))
    auth_status = "Logged In" if is_logged_in else "Not Logged In"
    print(f"Current Auth Status: {auth_status}")

    last_user_message = ""
    messages = state.get('messages', [])
    if messages and isinstance(messages[-1], HumanMessage):
        last_user_message = messages[-1].content
    else:
        print("L1 RAG Warning: No user message found. Routing to Support.")
        return {"next_action": "L2_SupportSupervisor", "current_task": "Initial State or Error", "error_message": "Missing User Input"}

    print(f"User Message: '{last_user_message}'")

    # Use the RAG utility function
    best_match_route, best_score = find_best_match(last_user_message, l1_route_names, l1_route_embeddings_np, confidence_threshold=0.3) # L1 threshold

    final_route = best_match_route
    if not final_route:
        print("L1 RAG: Low confidence or error in matching. Routing to Support.")
        final_route = "L2_SupportSupervisor" # Fallback if no good match
    elif final_route == "L2_AccountSupervisor" and not is_logged_in:
        print("L1 RAG Rule Applied: Account task requires login. Routing to Auth.")
        final_route = "L2_AuthSupervisor"

    print(f"L1 Final Routing Decision: {final_route}")
    return {
        "next_action": final_route,
        "current_task": last_user_message, # Pass the original message as the task
        "task_result": None,
        "error_message": None # Clear previous errors before routing
    }


# -----------------------------------------------------------------------------
# --- Level 2 Supervisors (RAG Based - using utility function) ---

def l2_auth_supervisor(state: AppState) -> dict:
    """Routes authentication tasks to specific L3 tools using RAG."""
    print("--- L2 Auth Supervisor (RAG Based) ---")
    task = state.get("current_task", "")
    print(f"Received Task: '{task}'")

    # Use RAG utility for L3 tool selection
    best_match_tool, best_score = find_best_match(task, l2_auth_tool_names, l2_auth_tool_embeddings_np, confidence_threshold=0.35) # Slightly higher threshold?

    next_action = best_match_tool
    if not next_action:
        # Fallback if RAG doesn't find a good match within Auth domain
        print("L2 Auth RAG: Low confidence or error. Routing to L2 Support for clarification/handoff.")
        next_action = "L2_SupportSupervisor" # Route to Support for ambiguous auth requests
        # Optionally set an error message or keep task
        state["error_message"] = "Unclear authentication request"

    print(f"L2 Auth Routing Decision: {next_action}")
    return {"next_action": next_action, "error_message": state.get("error_message")} # Pass along potential error message


def l2_account_supervisor(state: AppState) -> dict:
    """Routes account management tasks to specific L3 tools using RAG, with login check."""
    print("--- L2 Account Supervisor (RAG Based) ---")
    task = state.get("current_task", "")
    print(f"Received Task: '{task}'")

    # --- CRITICAL: Perform state check BEFORE RAG ---
    if not state.get("user_info"):
        print("L2 Account: Authentication required. Routing to L2 Auth Supervisor.")
        original_task = state.get("current_task", "your request")
        return {
            "next_action": "L2_AuthSupervisor",
            "current_task": f"login (required for: {original_task})", # Update task to reflect login need
            "error_message": "Authentication Required" # Signal error state
        }
    # --- End State Check ---

    # Use RAG utility for L3 tool selection (includes the special DISPLAY_DETAILS case)
    best_match_tool, best_score = find_best_match(task, l2_account_tool_names, l2_account_tool_embeddings_np, confidence_threshold=0.3) # Account threshold

    next_action = best_match_tool
    error_message = None # Start with no error for this stage

    if not next_action:
        # Fallback if RAG doesn't find a good match within Account domain
        print("L2 Account RAG: Low confidence or error. Routing to L2 Support.")
        next_action = "L2_SupportSupervisor"
        error_message = "Unclear account request"
    elif next_action == DISPLAY_DETAILS_ACTION:
        # Handle the special case directly within L2
        print("L2 Account: Handling display details action directly.")
        user_info = state["user_info"]
        details = (
            f"Login/Display Name: {user_info.get('name', 'N/A')}\n"
            f"Account Holder Name: {user_info.get('account_holder_name', 'N/A')}\n"
            f"Account Number: {user_info.get('account_number', 'N/A')}\n"
            f"Account Type: {user_info.get('account_type', 'N/A')}\n"
            f"Account ID: {user_info.get('account_id', 'N/A')}"
        )
        print(details + "\n")
        # Set result and end the turn
        return {"task_result": details, "next_action": END, "error_message": None}
    # else: proceed with the matched L3 tool node name

    print(f"L2 Account Routing Decision: {next_action}")
    return {"next_action": next_action, "error_message": error_message}


def l2_support_supervisor(state: AppState) -> dict:
    """Routes support tasks to specific L3 tools using RAG."""
    print("--- L2 Support Supervisor (RAG Based) ---")
    task = state.get("current_task", "")
    error_msg_in = state.get("error_message") # Check for errors passed to support
    print(f"Received Task: '{task}', Error In: '{error_msg_in}'")

    # --- Handle specific incoming errors BEFORE RAG ---
    if error_msg_in == "FAQ Not Found":
        print("L2 Support: Handling 'FAQ Not Found' error, forcing handoff.")
        # Clear the error as we're handling it
        return {"next_action": "L3_HumanHandoffNode", "error_message": None}
    if error_msg_in in ["Unknown Auth Task", "Unknown Account Task", "Unclear authentication request", "Unclear account request", "L1 Routing Failed", "Invalid L1 Routing Decision", "Invalid L2 Routing Decision", "L1 Embedding Error"]:
        print(f"L2 Support: Handling specific error '{error_msg_in}', forcing handoff.")
        # Keep error for context if needed by handoff? Or clear it? Let's clear for now.
        return {"next_action": "L3_HumanHandoffNode", "error_message": None}
    # --- End Error Handling ---

    # Use RAG utility for L3 tool selection (FAQ vs Handoff)
    # Use a lower threshold here, as we WANT to route somewhere, Handoff is the ultimate fallback
    best_match_tool, best_score = find_best_match(task, l2_support_tool_names, l2_support_tool_embeddings_np, confidence_threshold=0.25)

    next_action = best_match_tool
    if not next_action:
        # If RAG fails even here, default to Handoff
        print("L2 Support RAG: Low confidence or error. Defaulting to Human Handoff.")
        next_action = "L3_HumanHandoffNode"

    print(f"L2 Support Routing Decision: {next_action}")
    # Clear any non-critical incoming error message if we successfully routed via RAG
    final_error_message = None # Assume error handled unless set otherwise
    return {"next_action": next_action, "error_message": final_error_message}

# -----------------------------------------------------------------------------
# Graph Definition (Nodes are the same, logic inside L2 nodes changed)

# Conditional routing functions remain the same
def route_l1_decision(state: AppState) -> str:
    next_node = state.get("next_action")
    print(f"[Router] L1 Decision: Route to {next_node}")
    if next_node in ["L2_AuthSupervisor", "L2_AccountSupervisor", "L2_SupportSupervisor"]:
        return next_node
    else:
        print(f"[Router] L1 Warning/Error: Invalid next_action '{next_node}'. Defaulting to Support.")
        state["error_message"] = state.get("error_message") or f"Invalid L1 Routing Decision: {next_node}"
        return "L2_SupportSupervisor" # Route to support on invalid L1 decision

def route_l2_decision(state: AppState) -> str:
    # This router now receives L3 tool names OR L2 supervisor names (for fallback/reroute)
    next_node = state.get("next_action")
    print(f"[Router] L2 Decision: Route to {next_node}")

    # Define all valid destinations from an L2 supervisor
    valid_l3_nodes = l2_auth_tool_names + [name for name in l2_account_tool_names if name != DISPLAY_DETAILS_ACTION] + l2_support_tool_names
    valid_l2_reroutes = ["L2_AuthSupervisor", "L2_SupportSupervisor"] # Account supervisor handles its own login redirect

    if next_node in valid_l3_nodes:
        return next_node # Route to the determined L3 tool
    elif next_node in valid_l2_reroutes:
        return next_node # Reroute to another L2 supervisor (e.g., fallback from Auth/Account to Support)
    elif next_node == END:
         # This happens if L2 Account Supervisor handled details directly
         return END
    else:
        # This case should be less likely if L2 fallbacks work, but good safeguard
        print(f"[Router] L2 Warning: Invalid next_action '{next_node}' after L2 processing. Defaulting to Support Supervisor.")
        state["error_message"] = state.get("error_message") or f"Invalid L2 Routing Decision: {next_node}"
        # Avoid infinite loop: Route to Handoff instead of Support Supervisor if error persists
        # return "L3_HumanHandoffNode"
        return "L2_SupportSupervisor" # Let L2 Support handle via its error checks first


def route_after_tool(state: AppState) -> str:
    # This logic remains largely the same: handle tool errors or end the turn
    print(f"[Router] After Tool Execution...")
    error_message = state.get("error_message")
    next_action_forced_by_tool = state.get("next_action") # e.g., Logout sets user_info=None

    # If tool explicitly signals END (like human handoff)
    if next_action_forced_by_tool == END:
        print("Routing to END (forced by tool).")
        return END

    # If tool execution resulted in an error
    if error_message:
        print(f"Error detected after tool: {error_message}.")
        # Route specific errors for potential retry/correction
        if error_message in ["Authentication Required", "Authentication Failed", "Account Data Mismatch"]:
             print("Authentication error detected. Routing to L2 Auth Supervisor.")
             # Ensure user_info is cleared on failure/mismatch
             if error_message != "Authentication Required":
                 state["user_info"] = None
             state["error_message"] = None # Clear error before retrying auth flow
             return "L2_AuthSupervisor"
        elif error_message == "FAQ Not Found":
             print("FAQ not found error. Routing to L2 Support Supervisor for handoff.")
             # Keep error message for L2 Support to see
             return "L2_SupportSupervisor"
        # Handle other potentially recoverable errors?
        # elif error_message in ["Invalid New Login Name", "Invalid New Account Holder Name", "Missing Signup Field(s)"]:
        #     print(f"Input error: {error_message}. Ending turn, user needs to retry.")
        #     return END # End turn, main loop shows error
        else:
             # For unhandled errors from tools, end the turn.
             print("Ending turn due to unhandled tool error.")
             return END
    else:
        # Successful tool execution, end the current turn.
        print("Tool executed successfully. Ending current turn.")
        return END


# Build the graph (nodes and edges are defined the same way, using the updated functions)
builder = StateGraph(AppState)

# Add Nodes (using the RAG L1 and RAG L2 supervisors)
builder.add_node("L1_Supervisor", l1_main_supervisor_rag)
builder.add_node("L2_AuthSupervisor", l2_auth_supervisor)
builder.add_node("L2_AccountSupervisor", l2_account_supervisor)
builder.add_node("L2_SupportSupervisor", l2_support_supervisor)

# L3 Tool Nodes (unchanged definitions)
builder.add_node("L3_LoginToolNode", login_tool_node)
builder.add_node("L3_SignupToolNode", signup_tool_node)
builder.add_node("L3_PasswordResetToolNode", password_reset_tool_node)
builder.add_node("L3_LogoutToolNode", logout_tool_node)
builder.add_node("L3_CheckBalanceToolNode", check_balance_tool_node)
builder.add_node("L3_GetHistoryToolNode", get_history_tool_node)
builder.add_node("L3_LoginNameUpdateToolNode", login_name_update_tool_node)
builder.add_node("L3_AccountHolderNameUpdateToolNode", account_holder_name_update_tool_node)
builder.add_node("L3_FAQToolNode", faq_tool_node)
builder.add_node("L3_HumanHandoffNode", human_handoff_node)

# Define Edges
builder.add_edge(START, "L1_Supervisor")

# L1 to L2 routing (conditional based on L1 output)
builder.add_conditional_edges(
    "L1_Supervisor",
    route_l1_decision,
    {"L2_AuthSupervisor": "L2_AuthSupervisor", "L2_AccountSupervisor": "L2_AccountSupervisor", "L2_SupportSupervisor": "L2_SupportSupervisor"}
)

# L2 to L3/L2/END routing (conditional based on L2 output)
L3_TOOL_NODE_NAMES = valid_l3_nodes = list(set(l2_auth_tool_names + [name for name in l2_account_tool_names if name != DISPLAY_DETAILS_ACTION] + l2_support_tool_names))
L2_SUPERVISOR_NODE_NAMES = ["L2_AuthSupervisor", "L2_AccountSupervisor", "L2_SupportSupervisor"]

l2_conditional_map = {node: node for node in L3_TOOL_NODE_NAMES} # Map L3 tool names
l2_conditional_map["L2_AuthSupervisor"] = "L2_AuthSupervisor"     # Map reroute to Auth
l2_conditional_map["L2_SupportSupervisor"] = "L2_SupportSupervisor" # Map reroute to Support
l2_conditional_map[END] = END                                     # Allow L2 direct END

for supervisor_node in L2_SUPERVISOR_NODE_NAMES:
    builder.add_conditional_edges(
        supervisor_node,
        route_l2_decision, # This router reads the 'next_action' decided by the L2 RAG supervisor
        l2_conditional_map
    )

# Routing after L3 tools execute (conditional based on tool output/errors)
after_tool_map = {
    "L2_AuthSupervisor": "L2_AuthSupervisor",     # Reroute to Auth on auth errors
    "L2_SupportSupervisor": "L2_SupportSupervisor", # Reroute to Support on FAQ error
    END: END                                      # Default path for success or unhandled errors
}
for tool_node in L3_TOOL_NODE_NAMES:
     builder.add_conditional_edges(
         tool_node,
         route_after_tool,
         after_tool_map
     )

# Compile the graph
try:
    graph = builder.compile()
    print("\nGraph compiled successfully (using Langchain RAG L1 & L2 Supervisors)!")
except Exception as e:
    print(f"\nFatal Error: Graph compilation failed: {e}")
    traceback.print_exc()
    exit(1)

# Visualize (Optional)
try:
    output_filename = "banking_agent_graph_v4_full_rag.png"
    graph.get_graph().draw_mermaid_png(output_file_path=output_filename)
    print(f"Graph visualization saved to {output_filename}")
except ImportError:
    print("Install pygraphviz and graphviz to visualize the graph: pip install pygraphviz")
except Exception as e:
    print(f"Warning: Could not generate graph visualization: {e}")

# -----------------------------------------------------------------------------
# Main conversation loop (unchanged from previous RAG version)
# -----------------------------------------------------------------------------
def main():
    print("\n=== Welcome to the Multi-Level Banking Assistant (v4 - Full RAG Routing) ===")
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
        login_display_name = current_state.get("user_info", {}).get("name") if current_state.get("user_info") else None
        auth_display = f"(Logged in as: {login_display_name})" if login_display_name else "(Not Logged In)"
        user_input = input(f"\nYou {auth_display}: ")

        if 'quit' in user_input.lower() or 'exit' in user_input.lower():
            print("Banking Assistant: Goodbye!")
            break

        current_messages = current_state.get('messages', [])
        if not isinstance(current_messages, list):
             current_messages = list(current_messages)
        current_messages.append(HumanMessage(content=user_input))
        current_state['messages'] = current_messages

        # --- Clear state for the new turn ---
        # Keep user_info, clear the rest that are turn-specific
        current_state['task_result'] = None
        current_state['error_message'] = None
        current_state['next_action'] = None # MUST be cleared before graph invocation
        current_state['current_task'] = None # L1 will set this

        print("\nAssistant Processing...")
        try:
            # Graph stream execution
            for event in graph.stream(current_state, {"recursion_limit": 25}):
                node_name = list(event.keys())[0]
                node_output = event[node_name]
                # Optional: Reduce print verbosity for RAG steps if desired
                # print(f"--- Event: Node '{node_name}' Output: {node_output} ---")
                current_state.update(node_output) # Update state regardless of printing

            # --- After stream finishes for this input ---
            final_task_result = current_state.get("task_result")
            final_error_message = current_state.get("error_message")

            # Display final result or error for the turn
            if final_task_result:
                 print(f"\nAssistant: {final_task_result}")
            elif final_error_message:
                 # Provide more guidance on error
                 if final_error_message == "Authentication Required":
                      print(f"\nAssistant: Please log in first to complete your request.")
                 elif final_error_message in ["Authentication Failed", "Account Not Found", "Email Exists"]:
                      print(f"\nAssistant: There was an issue with authentication: {final_error_message}. Please try again.")
                 elif "Invalid" in final_error_message or "Missing" in final_error_message:
                     print(f"\nAssistant: There was an input error: {final_error_message}. Please try again.")
                 else:
                      print(f"\nAssistant: Sorry, I encountered an issue: {final_error_message}. Please try asking differently or ask for help.")

            else:
                 # If graph ended without a result or error message (e.g., details displayed directly)
                 if current_state.get("next_action") == END and not final_task_result and not final_error_message:
                     # This case is less likely now unless a tool/node returns END without setting task_result
                     # It could happen if L2 Account Supervisor displayed details
                      print("\nAssistant: Request completed.") # Generic completion
                 else:
                     # Graph ended unexpectedly? Or waiting for next input.
                     print("\nAssistant: How else can I assist you today?")


        except Exception as e:
            print(f"\n--- Critical Error during graph execution ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            traceback.print_exc()
            print("\nAssistant: I've encountered a critical system error. Please restart the conversation or try again later.")
            break # Stop processing on critical errors

if __name__ == "__main__":
    main()