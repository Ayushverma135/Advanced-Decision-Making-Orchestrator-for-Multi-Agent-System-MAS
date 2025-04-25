# --- Imports (unchanged) ---
import random
import string
import uuid
from typing import TypedDict, List, Optional, Sequence, Annotated, Tuple # Added Tuple
import torch # Still useful for device detection
import traceback # For error printing
import re # For keyword matching potentially

# --- Imports for Langchain Embedding approach (unchanged) ---
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("langchain-huggingface library loaded successfully.")
except ImportError:
    print("Fatal Error: langchain-huggingface library not found.")
    print("Please install it: pip install langchain-huggingface")
    exit(1)

# --- Imports for Similarity Calculation (unchanged) ---
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
# ... (Keep all this code) ...
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
    # Optional: Add state to hold suggestions if needed for fallback
    # suggested_actions: Optional[List[str]]

# -----------------------------------------------------------------------------
# Level 3: Specialist Tools / Nodes (unchanged)
# ... (Keep all L3 tool nodes) ...
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
# --- Hybrid RAG Setup ---

# 1. Define Hybrid Search Parameters
HYBRID_ALPHA = 0.8 # Weight for keyword score (0.0 to 1.0)
DIFF_THRESHOLD = 0.01 # Min score difference between top 2 for automatic routing
MIN_SCORE_THRESHOLD = 0.04 # Minimum hybrid score to consider a match valid at all

# 2. Initialize Embedding Model (unchanged)
# ... (embedding model loading code remains the same) ...
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

# 3. Define Route/Tool Data (Descriptions AND Keywords - unchanged)
# ... (definitions for l1_route, l2_auth, l2_account, l2_support remain the same) ...
# --- L1 Supervisor Data ---
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
        ),
        "keywords": ["log in", "login", "signin", "sign in", "authenticate", "access", "log out", "logout", "sign out", "exit", "disconnect", "register", "signup", "sign up", "create account", "new user", "enroll", "password", "forgot", "reset", "change password", "update password", "locked out", "credentials"]

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
        ),
         "keywords": ["balance", "funds", "money", "history", "transaction", "statement", "activity", "spending", "payments", "deposits", "withdrawals", "details", "number", "account number", "id", "account id", "type", "account type", "update name", "change name", "correct name", "holder name", "legal name", "login name", "display name", "username", "profile name"]
    },
    {
        "name": "L2_SupportSupervisor",
        "description": (
            "Handles general bank information (FAQs), requests for human assistance, and fallback for unclear queries. "
            "Keywords: hours, opening times, opening hours, when are you open. "
            "Keywords: locations, address, where are branches, find branch. "
            "Keywords: contact, phone number, call us, email address, customer service number. "
            "Keywords: help, support, assistance, need help, problem, issue, error, complaint, feedback. "
            "Keywords: talk to someone, human agent, representative, speak to person. "
            "Focus: General bank operations, getting help with the service, or when the intent isn't clearly authentication or account data management."
        ),
        "keywords": ["help", "support", "assist", "assistance", "agent", "human", "person", "talk to", "speak to", "representative", "manager", "operator", "issue", "problem", "error", "complain", "complaint", "feedback", "confused", "frustrated", "stuck", "hours", "open", "opening", "closed", "times", "contact", "phone", "call", "email", "location", "address", "branch", "atm", "fees", "charges", "rates", "interest", "website", "app"]
    }
]
# --- L2 Auth Supervisor Data ---
l2_auth_tool_definitions = [
    {
        "name": "L3_LoginToolNode",
        "description": "Gain access using existing credentials.",
        "keywords": ["log in", "login", "signin", "sign in", "logon", "authenticate", "access", "enter", "credentials"]
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
DISPLAY_DETAILS_ACTION = "L2_DISPLAY_DETAILS" # Special constant
l2_account_tool_definitions = [
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
l2_support_tool_definitions = [
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

# 4. Pre-compute Embeddings (unchanged)
# ... (compute_embeddings function and calls remain the same) ...
def compute_embeddings(definitions: List[dict]) -> tuple[List[str], List[List[str]], np.ndarray]:
    """Extracts names, keywords, computes embeddings for descriptions."""
    names = [d["name"] for d in definitions]
    descriptions = [d["description"] for d in definitions]
    keywords = [[kw.lower() for kw in d.get("keywords", [])] for d in definitions]
    try:
        embeddings_np = np.array(embedding_model.embed_documents(descriptions))
        print(f"Computed embeddings for {len(names)} items, shape: {embeddings_np.shape}")
    except Exception as e:
        print(f"Error computing embeddings: {e}")
        traceback.print_exc()
        raise e
    return names, keywords, embeddings_np

print("Computing embeddings for all levels...")
try:
    l1_route_names, l1_route_keywords, l1_route_embeddings_np = compute_embeddings(l1_route_definitions)
    l2_auth_tool_names, l2_auth_tool_keywords, l2_auth_tool_embeddings_np = compute_embeddings(l2_auth_tool_definitions)
    l2_account_tool_names, l2_account_tool_keywords, l2_account_tool_embeddings_np = compute_embeddings(l2_account_tool_definitions)
    l2_support_tool_names, l2_support_tool_keywords, l2_support_tool_embeddings_np = compute_embeddings(l2_support_tool_definitions)
    print("All embeddings computed successfully.")
except Exception as e:
    print("Fatal error during embedding computation. Exiting.")
    exit(1)


# 5. Keyword Scoring Function (Jaccard - unchanged)
# ... (keyword_similarity function remains the same) ...
# 5. Keyword Scoring Function (Reverted to Substring Check)
def calculate_keyword_score(query: str, keywords: List[str]) -> float:
    """Calculates a normalized keyword match score based on substring presence."""
    if not keywords:
        return 0.0
    query_lower = query.lower()
    # Count how many keywords are present as substrings
    match_count = sum(1 for kw in keywords if kw in query_lower)
    # Normalize score (proportion of keywords found) - simple approach
    score = float(match_count) / float(len(keywords))
    # Alternative: Boost significantly if *any* keyword is found
    # score = 1.0 if match_count > 0 else 0.0
    return min(score, 1.0) # Cap score at 1.0

# 6. Hybrid RAG Utility Function (MODIFIED to return top_k)
def get_top_k_hybrid_matches(
    query: str,
    names: List[str],
    keywords_list: List[List[str]],
    embeddings: np.ndarray,
    k: int = 2, # Number of top results to return
    alpha: float = HYBRID_ALPHA,
    min_score_threshold: float = MIN_SCORE_THRESHOLD # Minimum score to even consider
) -> List[Tuple[str, float]]: # Returns list of (name, score) tuples
    """Calculates hybrid scores and returns the top k matches above a minimum threshold."""
    if not query or not query.strip():
        print("[Hybrid RAG Util] Warning: Empty query provided.")
        return []

    num_options = len(names)
    if num_options == 0 or len(keywords_list) != num_options or embeddings.shape[0] != num_options:
        print("[Hybrid RAG Util] Error: Mismatched input list lengths.")
        return []

    results = []
    try:
        # --- Vector Score ---
        query_embedding_list = embedding_model.embed_query(query)
        query_embedding_np = np.array(query_embedding_list).reshape(1, -1)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        vector_scores = cosine_similarity(query_embedding_np, embeddings)[0]

        # --- Keyword & Hybrid Scores ---
        all_hybrid_scores = []
        keyword_scores_debug = {}
        vector_scores_debug = {}

        for i in range(num_options):
            keyword_score = calculate_keyword_score(query, keywords_list[i])
            vector_score = vector_scores[i]
            vector_score_clipped = max(0, vector_score)

            hybrid_score = (alpha * keyword_score) + ((1 - alpha) * vector_score_clipped)
            all_hybrid_scores.append((names[i], hybrid_score)) # Store name and score

            keyword_scores_debug[names[i]] = keyword_score
            vector_scores_debug[names[i]] = vector_score

        # --- Sort and Filter ---
        # Sort by hybrid score descending
        all_hybrid_scores.sort(key=lambda x: x[1], reverse=True)

        # Filter by minimum threshold
        filtered_results = [(name, score) for name, score in all_hybrid_scores if score >= min_score_threshold]

        # --- Debugging Output ---
        print(f"[Hybrid RAG Util] Query: '{query}'")
        print(f"[Hybrid RAG Util] Keyword Scores (Jaccard): { {k: f'{v:.4f}' for k, v in keyword_scores_debug.items()} }")
        print(f"[Hybrid RAG Util] Vector Scores (Cosine):   { {k: f'{v:.4f}' for k, v in vector_scores_debug.items()} }")
        print(f"[Hybrid RAG Util] Sorted Hybrid Scores (alpha={alpha:.2f}): { [(n, f'{s:.4f}') for n, s in all_hybrid_scores] }")

        # Return top k results from the filtered list
        results = filtered_results[:k]
        print(f"[Hybrid RAG Util] Top {k} results above threshold {min_score_threshold}: {results}")
        return results

    except Exception as e:
        print(f"[Hybrid RAG Util] Error during hybrid matching: {e}")
        traceback.print_exc()
        return []


# -----------------------------------------------------------------------------
# --- Level 1 Supervisor (Using Score Difference Logic) ---
def l1_main_supervisor_rag(state: AppState) -> dict:
    """Routes user requests to L2 supervisors using Hybrid RAG + Score Difference."""
    print("\n--- L1 Main Supervisor (Hybrid RAG - Diff Logic) ---")
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

    # Get top 2 results using the Hybrid RAG utility function
    top_matches = get_top_k_hybrid_matches(
        query=last_user_message,
        names=l1_route_names,
        keywords_list=l1_route_keywords,
        embeddings=l1_route_embeddings_np,
        k=2,
        min_score_threshold=MIN_SCORE_THRESHOLD # Apply minimal filtering
    )

    next_action = "L2_SupportSupervisor" # Default fallback
    error_message = None

    if not top_matches:
        print("L1 Hybrid RAG: No relevant match found. Routing to Support.")
        error_message = "Could not understand request"
    elif len(top_matches) == 1:
        # Only one match above minimum threshold, route to it
        best_match_name, best_score = top_matches[0]
        print(f"L1 Hybrid RAG: Only one relevant match found ({best_match_name}, Score: {best_score:.4f}). Routing.")
        next_action = best_match_name
    else:
        # Compare top two matches
        top_name, top_score = top_matches[0]
        second_name, second_score = top_matches[1]
        diff = top_score - second_score
        print(f"L1 Hybrid RAG: Top score={top_score:.4f} ({top_name}), Second score={second_score:.4f} ({second_name}), Diff={diff:.4f}")

        if diff > DIFF_THRESHOLD:
            print(f"L1 Hybrid RAG: High confidence (Diff > {DIFF_THRESHOLD}). Routing to {top_name}.")
            next_action = top_name
        else:
            print(f"L1 Hybrid RAG: Low confidence / Ambiguous (Diff <= {DIFF_THRESHOLD}). Routing to Support.")
            next_action = "L2_SupportSupervisor"
            error_message = f"Ambiguous request (Similar to {top_name} and {second_name})" # Add context

    # Apply login check override if routing to Account
    if next_action == "L2_AccountSupervisor" and not is_logged_in:
        print("L1 RAG Rule Applied: Account task requires login. Routing to Auth.")
        next_action = "L2_AuthSupervisor"
        error_message = None # Override ambiguity error if login is required

    print(f"L1 Final Routing Decision: {next_action}")
    return {
        "next_action": next_action,
        "current_task": last_user_message,
        "task_result": None,
        "error_message": error_message
    }

# -----------------------------------------------------------------------------
# --- Level 2 Supervisors (Using Score Difference Logic) ---

def l2_auth_supervisor(state: AppState) -> dict:
    """Routes authentication tasks using Hybrid RAG + Score Difference."""
    print("--- L2 Auth Supervisor (Hybrid RAG - Diff Logic) ---")
    task = state.get("current_task", "")
    print(f"Received Task: '{task}'")

    top_matches = get_top_k_hybrid_matches(
        query=task,
        names=l2_auth_tool_names,
        keywords_list=l2_auth_tool_keywords,
        embeddings=l2_auth_tool_embeddings_np,
        k=2,
        min_score_threshold=MIN_SCORE_THRESHOLD
    )

    next_action = "L2_SupportSupervisor" # Default fallback within Auth -> Support
    error_message = state.get("error_message") # Preserve incoming error

    if not top_matches:
        print("L2 Auth Hybrid RAG: No relevant match found. Routing to Support.")
        error_message = error_message or "Unclear authentication request"
    elif len(top_matches) == 1:
        best_match_name, best_score = top_matches[0]
        print(f"L2 Auth Hybrid RAG: Only one relevant match found ({best_match_name}, Score: {best_score:.4f}). Routing.")
        next_action = best_match_name
        error_message = None # Clear error if we found a specific tool
    else:
        top_name, top_score = top_matches[0]
        second_name, second_score = top_matches[1]
        diff = top_score - second_score
        print(f"L2 Auth Hybrid RAG: Top score={top_score:.4f} ({top_name}), Second score={second_score:.4f} ({second_name}), Diff={diff:.4f}")

        if diff > DIFF_THRESHOLD:
            print(f"L2 Auth Hybrid RAG: High confidence (Diff > {DIFF_THRESHOLD}). Routing to {top_name}.")
            next_action = top_name
            error_message = None # Clear error if we found a specific tool
        else:
            print(f"L2 Auth Hybrid RAG: Low confidence / Ambiguous (Diff <= {DIFF_THRESHOLD}). Routing to Support.")
            next_action = "L2_SupportSupervisor"
            error_message = f"Ambiguous authentication request (Similar to {top_name} and {second_name})"

    print(f"L2 Auth Routing Decision: {next_action}")
    return {"next_action": next_action, "error_message": error_message}


def l2_account_supervisor(state: AppState) -> dict:
    """Routes account tasks using Hybrid RAG + Score Difference, with login check."""
    print("--- L2 Account Supervisor (Hybrid RAG - Diff Logic) ---")
    task = state.get("current_task", "")
    print(f"Received Task: '{task}'")

    # --- CRITICAL: Login Check ---
    if not state.get("user_info"):
        print("L2 Account: Authentication required. Routing to L2 Auth Supervisor.")
        return {
            "next_action": "L2_AuthSupervisor",
            "current_task": f"login (required for: {task or 'your request'})",
            "error_message": "Authentication Required"
        }
    # --- End Login Check ---

    top_matches = get_top_k_hybrid_matches(
        query=task,
        names=l2_account_tool_names,
        keywords_list=l2_account_tool_keywords,
        embeddings=l2_account_tool_embeddings_np,
        k=2,
        min_score_threshold=MIN_SCORE_THRESHOLD
    )

    next_action = "L2_SupportSupervisor" # Default fallback
    error_message = None

    if not top_matches:
        print("L2 Account Hybrid RAG: No relevant match found. Routing to Support.")
        error_message = "Unclear account request"
    elif len(top_matches) == 1:
        best_match_name, best_score = top_matches[0]
        print(f"L2 Account Hybrid RAG: Only one relevant match ({best_match_name}, Score: {best_score:.4f}). Routing.")
        next_action = best_match_name
    else:
        top_name, top_score = top_matches[0]
        second_name, second_score = top_matches[1]
        diff = top_score - second_score
        print(f"L2 Account Hybrid RAG: Top score={top_score:.4f} ({top_name}), Second score={second_score:.4f} ({second_name}), Diff={diff:.4f}")

        if diff > DIFF_THRESHOLD:
            print(f"L2 Account Hybrid RAG: High confidence (Diff > {DIFF_THRESHOLD}). Routing to {top_name}.")
            next_action = top_name
        else:
            print(f"L2 Account Hybrid RAG: Low confidence / Ambiguous (Diff <= {DIFF_THRESHOLD}). Routing to Support.")
            next_action = "L2_SupportSupervisor"
            error_message = f"Ambiguous account request (Similar to {top_name} and {second_name})"

    # Handle special display details action AFTER routing decision
    if next_action == DISPLAY_DETAILS_ACTION:
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
        return {"task_result": details, "next_action": END, "error_message": None} # End turn

    print(f"L2 Account Routing Decision: {next_action}")
    return {"next_action": next_action, "error_message": error_message}


def l2_support_supervisor(state: AppState) -> dict:
    """Routes support tasks using Hybrid RAG + Score Difference."""
    print("--- L2 Support Supervisor (Hybrid RAG - Diff Logic) ---")
    task = state.get("current_task", "")
    error_msg_in = state.get("error_message")
    print(f"Received Task: '{task}', Error In: '{error_msg_in}'")

    # --- Handle specific incoming errors BEFORE RAG ---
    if error_msg_in == "FAQ Not Found":
        print("L2 Support: Handling 'FAQ Not Found' error, forcing handoff.")
        return {"next_action": "L3_HumanHandoffNode", "error_message": None}
    # Expanded error check list
    specific_errors = [
        "Unknown Auth Task", "Unknown Account Task",
        "Unclear authentication request", "Unclear account request",
        "L1 Routing Failed", "Invalid L1 Routing Decision", "Invalid L2 Routing Decision",
        "L1 Embedding Error", "Could not understand request", # From L1 fallback
        "Ambiguous request", # From L1 ambiguity
        "Ambiguous authentication request", # From L2 Auth ambiguity
        "Ambiguous account request" # From L2 Account ambiguity
    ]
    # Check if any part of the error message matches known patterns
    if error_msg_in and any(err_pattern in error_msg_in for err_pattern in specific_errors):
        print(f"L2 Support: Handling specific error/ambiguity '{error_msg_in}', forcing handoff.")
        return {"next_action": "L3_HumanHandoffNode", "error_message": None} # Clear error before handoff
    # --- End Error Handling ---

    # Use RAG for FAQ vs Handoff
    top_matches = get_top_k_hybrid_matches(
        query=task,
        names=l2_support_tool_names,
        keywords_list=l2_support_tool_keywords,
        embeddings=l2_support_tool_embeddings_np,
        k=2,
        min_score_threshold=MIN_SCORE_THRESHOLD # Lower threshold okay here?
    )

    # Default to handoff if no match or ambiguity
    next_action = "L3_HumanHandoffNode"

    if not top_matches:
        print("L2 Support Hybrid RAG: No relevant match found. Defaulting to Human Handoff.")
    elif len(top_matches) == 1:
        best_match_name, best_score = top_matches[0]
        print(f"L2 Support Hybrid RAG: Only one relevant match ({best_match_name}, Score: {best_score:.4f}). Routing.")
        next_action = best_match_name
    else:
        top_name, top_score = top_matches[0]
        second_name, second_score = top_matches[1]
        diff = top_score - second_score
        print(f"L2 Support Hybrid RAG: Top score={top_score:.4f} ({top_name}), Second score={second_score:.4f} ({second_name}), Diff={diff:.4f}")

        # Even if difference is small, if top is FAQ, try it? Otherwise handoff.
        # Or stick to diff threshold logic? Let's stick to diff threshold.
        if diff > DIFF_THRESHOLD:
            print(f"L2 Support Hybrid RAG: High confidence (Diff > {DIFF_THRESHOLD}). Routing to {top_name}.")
            next_action = top_name
        else:
            print(f"L2 Support Hybrid RAG: Low confidence / Ambiguous (Diff <= {DIFF_THRESHOLD}). Defaulting to Human Handoff.")
            next_action = "L3_HumanHandoffNode" # Fallback is handoff

    print(f"L2 Support Routing Decision: {next_action}")
    return {"next_action": next_action, "error_message": None} # Clear error if handled


# -----------------------------------------------------------------------------
# Graph Definition & Main Loop (Completely unchanged from the previous version)
# The structure remains the same; only the internal logic of the supervisor nodes changed.
# -----------------------------------------------------------------------------
# Graph Definition (Structurally the same, nodes and edges point to correct functions)

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
    # Get unique L3 tool names from all lists
    valid_l3_nodes_set = set(l2_auth_tool_names +
                           [name for name in l2_account_tool_names if name != DISPLAY_DETAILS_ACTION] +
                           l2_support_tool_names)
    valid_l3_nodes = list(valid_l3_nodes_set)
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

# Add Nodes (using the Hybrid RAG L1 and L2 supervisors)
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
# Get unique L3 tool names again for the map
L3_TOOL_NODE_NAMES = list(set(l2_auth_tool_names +
                           [name for name in l2_account_tool_names if name != DISPLAY_DETAILS_ACTION] +
                           l2_support_tool_names))
L2_SUPERVISOR_NODE_NAMES = ["L2_AuthSupervisor", "L2_AccountSupervisor", "L2_SupportSupervisor"]

l2_conditional_map = {node: node for node in L3_TOOL_NODE_NAMES} # Map L3 tool names
l2_conditional_map["L2_AuthSupervisor"] = "L2_AuthSupervisor"     # Map reroute to Auth
l2_conditional_map["L2_SupportSupervisor"] = "L2_SupportSupervisor" # Map reroute to Support
l2_conditional_map[END] = END                                     # Allow L2 direct END

for supervisor_node in L2_SUPERVISOR_NODE_NAMES:
    builder.add_conditional_edges(
        supervisor_node,
        route_l2_decision, # This router reads the 'next_action' decided by the L2 Hybrid RAG supervisor
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
    print("\nGraph compiled successfully (using Langchain Hybrid RAG - Diff Logic)!")
except Exception as e:
    print(f"\nFatal Error: Graph compilation failed: {e}")
    traceback.print_exc()
    exit(1)

# Visualize (Optional)
try:
    output_filename = "banking_agent_graph_v6_hybrid_diff_rag.png"
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
    print("\n=== Welcome to the Multi-Level Banking Assistant (v6 - Hybrid RAG - Diff Logic) ===")
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
        current_state['task_result'] = None
        current_state['error_message'] = None
        current_state['next_action'] = None
        current_state['current_task'] = None

        print("\nAssistant Processing...")
        try:
            # Graph stream execution
            for event in graph.stream(current_state, {"recursion_limit": 25}):
                node_name = list(event.keys())[0]
                node_output = event[node_name]
                # Optional: Reduce print verbosity
                # print(f"--- Event: Node '{node_name}' Output: {node_output} ---")
                current_state.update(node_output)

            # --- After stream finishes ---
            final_task_result = current_state.get("task_result")
            final_error_message = current_state.get("error_message")

            # Display final result or error
            if final_task_result:
                 print(f"\nAssistant: {final_task_result}")
            elif final_error_message:
                 # Provide more context for ambiguity errors
                 if "Ambiguous request" in final_error_message:
                     print(f"\nAssistant: I'm not sure exactly what you need. {final_error_message}. Could you please clarify or ask for 'help'?")
                 elif final_error_message == "Authentication Required":
                      print(f"\nAssistant: Please log in first to complete your request.")
                 elif final_error_message in ["Authentication Failed", "Account Not Found", "Email Exists"]:
                      print(f"\nAssistant: There was an issue with authentication: {final_error_message}. Please try again.")
                 elif "Invalid" in final_error_message or "Missing" in final_error_message:
                     print(f"\nAssistant: There was an input error: {final_error_message}. Please try again.")
                 else: # Generic fallback error message
                      print(f"\nAssistant: Sorry, I encountered an issue: {final_error_message}. Please try asking differently or ask for 'help'.")

            else:
                 if current_state.get("next_action") == END and not final_task_result and not final_error_message:
                     print("\nAssistant: Request completed.")
                 else:
                     # This might happen if a fallback route leads to a supervisor that then ends the turn without action
                     print("\nAssistant: How else can I assist you today?")


        except Exception as e:
             # ... (critical error handling unchanged) ...
            print(f"\n--- Critical Error during graph execution ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            traceback.print_exc()
            print("\nAssistant: I've encountered a critical system error. Please restart the conversation or try again later.")
            break

if __name__ == "__main__":
    main()