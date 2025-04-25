# --- Imports (unchanged) ---
# --- Imports for Langchain Embedding approach (unchanged) ---
# --- Imports for Similarity Calculation (unchanged) ---
# --- Langchain Core & Langgraph Imports (unchanged) ---
import random
import string
import uuid
from typing import TypedDict, List, Optional, Sequence, Annotated, Tuple
import torch
import traceback
import re
import time
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# -----------------------------------------------------------------------------
# Helper functions, DBs (unchanged)
# ... (code) ...
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
# State Definition (MODIFIED)
# -----------------------------------------------------------------------------
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
    suggested_choices: Optional[List[Tuple[str, str]]] # For clarification node
     # --- NEW field to pass L2 results to L3 ---
    top_matches_from_l2: Optional[List[Tuple[str, float]]] # List of (name, score)

# -----------------------------------------------------------------------------
# --- Constants and RAG Setup ---
# -----------------------------------------------------------------------------

# 1. Define Hybrid Search Parameters
HYBRID_ALPHA = 0.8
DIFF_THRESHOLD = 0.2 # <<< Ambiguity threshold set by user
MIN_SCORE_THRESHOLD = 0.04

# 2. Initialize Embedding Model (unchanged)
# ... (embedding model loading code) ...
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
# ... (definitions for l1_route, l2_auth, l2_account, l2_support) ...
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
        # Added "sign in" as a single keyword
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
# ... (compute_embeddings function and calls) ...
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
    # --- Create global lookup for descriptions ---
    all_definitions_list = (l1_route_definitions + l2_auth_tool_definitions +
                         l2_account_tool_definitions + l2_support_tool_definitions)
    description_lookup = {d["name"]: d["description"] for d in all_definitions_list}
except Exception as e:
    print("Fatal error during embedding computation or lookup creation. Exiting.")
    exit(1)


# 5. Keyword Scoring Function (Substring Check - unchanged)
# ... (calculate_keyword_score function) ...
def calculate_keyword_score(query: str, keywords: List[str]) -> float:
    """Calculates a normalized keyword match score based on substring presence."""
    if not keywords:
        return 0.0
    query_lower = query.lower()
    match_count = sum(1 for kw in keywords if kw in query_lower)
    score = float(match_count) / float(len(keywords)) if keywords else 0.0
    return min(score, 1.0)

# 6. Hybrid RAG Utility Function (Unchanged - returns top k)
# ... (get_top_k_hybrid_matches function) ...
def get_top_k_hybrid_matches(
    query: str,
    names: List[str],
    keywords_list: List[List[str]],
    embeddings: np.ndarray,
    k: int = 2,
    alpha: float = HYBRID_ALPHA,
    min_score_threshold: float = MIN_SCORE_THRESHOLD
) -> List[Tuple[str, float]]:
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
        query_embedding_list = embedding_model.embed_query(query)
        query_embedding_np = np.array(query_embedding_list).reshape(1, -1)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        vector_scores = cosine_similarity(query_embedding_np, embeddings)[0]

        all_hybrid_scores = []
        keyword_scores_debug = {}
        vector_scores_debug = {}
        for i in range(num_options):
            keyword_score = calculate_keyword_score(query, keywords_list[i])
            vector_score = vector_scores[i]
            vector_score_clipped = max(0, vector_score)
            hybrid_score = (alpha * keyword_score) + ((1 - alpha) * vector_score_clipped)
            all_hybrid_scores.append((names[i], hybrid_score))
            keyword_scores_debug[names[i]] = keyword_score
            vector_scores_debug[names[i]] = vector_score

        all_hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        filtered_results = [(name, score) for name, score in all_hybrid_scores if score >= min_score_threshold]

        print(f"[Hybrid RAG Util] Query: '{query}'")
        print(f"[Hybrid RAG Util] Keyword Scores (Substring): { {k: f'{v:.4f}' for k, v in keyword_scores_debug.items()} }")
        print(f"[Hybrid RAG Util] Vector Scores (Cosine):   { {k: f'{v:.4f}' for k, v in vector_scores_debug.items()} }")
        print(f"[Hybrid RAG Util] Sorted Hybrid Scores (alpha={alpha:.2f}): { [(n, f'{s:.4f}') for n, s in all_hybrid_scores] }")

        results = filtered_results[:k]
        print(f"[Hybrid RAG Util] Top {k} results above threshold {min_score_threshold}: {results}")
        return results
    except Exception as e:
        print(f"[Hybrid RAG Util] Error during hybrid matching: {e}")
        traceback.print_exc()
        return []

# 7. Helper to get description (unchanged)
# ... (get_node_description function using description_lookup) ...
def get_node_description(node_name: str) -> str:
    """Looks up a brief description for a given node name."""
    return description_lookup.get(node_name, "Perform this action") # Default description


# 8. Clarification Node (Unchanged)
# ... (ask_for_clarification_node function) ...
def ask_for_clarification_node(state: AppState) -> dict:
    """Presents choices to the user and gets their selection."""
    print("--- Clarification Needed ---")
    choices = state.get("suggested_choices")
    if not choices:
        print("Error: No choices provided for clarification. Routing to support.")
        return {"next_action": "L2_SupportSupervisor", "error_message": "Internal error: Missing clarification choices", "suggested_choices": None, "top_matches_from_l2": None}

    print("I'm not sure exactly what you mean. Did you want to:")
    choice_map = {}
    for i, (node_name, description) in enumerate(choices):
        label = node_name.replace("L2_", "").replace("L3_", "").replace("Supervisor", "").replace("ToolNode", "")
        print(f"  {i+1}. {label} ({description})")
        choice_map[str(i+1)] = node_name # Map number string to node name

    while True:
        user_choice_str = input(f"Please enter the number of your choice (1-{len(choices)}): ")
        chosen_node_name = choice_map.get(user_choice_str)
        if chosen_node_name:
            print(f"Okay, proceeding with: {chosen_node_name}")
            # Clear choices and matches, set the next action based on user selection
            return {"next_action": chosen_node_name, "suggested_choices": None, "error_message": None, "top_matches_from_l2": None}
        else:
            print(f"Invalid choice. Please enter a number between 1 and {len(choices)}.")


# -----------------------------------------------------------------------------
# --- Level 1 Supervisor (MODIFIED - Only passes results) ---
# -----------------------------------------------------------------------------
def l1_main_supervisor_rag(state: AppState) -> dict:
    """Gets top L2 routes using Hybrid RAG and passes results."""
    print("\n--- L1 Main Supervisor (Passes Results) ---")
    is_logged_in = bool(state.get('user_info'))
    auth_status = "Logged In" if is_logged_in else "Not Logged In"
    print(f"Current Auth Status: {auth_status}")
    last_user_message = ""
    messages = state.get('messages', [])
    if messages and isinstance(messages[-1], HumanMessage):
        last_user_message = messages[-1].content
    else:
        print("L1 RAG Warning: No user message found. Routing to Support.")
        return {"next_action": "L2_SupportSupervisor", "current_task": "Initial State or Error", "error_message": "Missing User Input", "top_matches_from_l2": None}
    print(f"User Message: '{last_user_message}'")

    # Get top matches
    top_matches = get_top_k_hybrid_matches(
        query=last_user_message,
        names=l1_route_names,
        keywords_list=l1_route_keywords,
        embeddings=l1_route_embeddings_np,
        k=3, # Get top 3
        min_score_threshold=MIN_SCORE_THRESHOLD
    )

    next_action = "L2_SupportSupervisor" # Default fallback
    final_top_matches = None

    if top_matches:
        # Always route to the top match initially
        next_action = top_matches[0][0]
        final_top_matches = top_matches # Pass results down
        print(f"L1 Hybrid RAG: Top match is {next_action}. Passing results.")
    else:
        print("L1 Hybrid RAG: No relevant match found. Routing to Support.")
        # Error message will be set implicitly if needed by L2 support

    # Apply login check override AFTER determining the potential next_action
    if next_action == "L2_AccountSupervisor" and not is_logged_in:
        print("L1 RAG Rule Applied: Account task requires login. Routing to Auth.")
        next_action = "L2_AuthSupervisor"
        final_top_matches = None # Don't pass matches if redirecting for login

    print(f"L1 Final Routing Decision: {next_action}")
    # Pass the top_matches list in the state
    return {
        "next_action": next_action,
        "current_task": last_user_message,
        "top_matches_from_l2": final_top_matches, # Pass matches
        "task_result": None,
        "error_message": None, # L2/L3 will handle ambiguity
        "suggested_choices": None # Not decided here
    }

# -----------------------------------------------------------------------------
# --- Level 2 Supervisors (MODIFIED - Only passes results) ---
# -----------------------------------------------------------------------------

# Helper function for L2 Supervisors
def l2_supervisor_router(
    state: AppState,
    supervisor_name: str,
    tool_names: List[str],
    tool_keywords: List[List[str]],
    tool_embeddings: np.ndarray,
    fallback_node: str = "L2_SupportSupervisor" # Default fallback
) -> dict:
    """Generic L2 logic: finds top L3 tools, passes results."""
    print(f"\n--- {supervisor_name} (Passes Results) ---")
    task = state.get("current_task", "")
    print(f"Received Task: '{task}'")

    # --- Login Check (Specific to Account Supervisor) ---
    if supervisor_name == "L2_AccountSupervisor" and not state.get("user_info"):
        print("L2 Account: Authentication required. Routing to L2 Auth Supervisor.")
        return {
            "next_action": "L2_AuthSupervisor",
            "current_task": f"login (required for: {task or 'your request'})",
            "error_message": "Authentication Required",
            "top_matches_from_l2": None # Clear matches
        }
    # --- End Login Check ---

    top_matches = get_top_k_hybrid_matches(
        query=task,
        names=tool_names,
        keywords_list=tool_keywords,
        embeddings=tool_embeddings,
        k=3,
        min_score_threshold=MIN_SCORE_THRESHOLD
    )

    next_action = fallback_node # Default fallback if no match
    final_top_matches = None
    error_message = state.get("error_message") # Preserve incoming error?

    if top_matches:
        # Route to the top L3 match
        next_action = top_matches[0][0]
        final_top_matches = top_matches # Pass results to L3
        print(f"{supervisor_name} Hybrid RAG: Top match is {next_action}. Passing results.")
        # If we found a match, potentially clear incoming error? Maybe not, let L3 decide.
        # error_message = None
    else:
        print(f"{supervisor_name} Hybrid RAG: No relevant match found. Routing to fallback {fallback_node}.")
        # Set error if none exists
        error_message = error_message or f"Unclear request for {supervisor_name}"

    # Handle Account Supervisor's special display action case
    if supervisor_name == "L2_AccountSupervisor" and next_action == DISPLAY_DETAILS_ACTION:
         print("L2 Account: Handling display details action directly.")
         # ... (logic to display details - MUST RETURN **END** and clear matches) ...
         user_info = state["user_info"]
         details = (f"Login/Display Name: {user_info.get('name', 'N/A')}\n"
                    f"Account Holder Name: {user_info.get('account_holder_name', 'N/A')}\n"
                    f"Account Number: {user_info.get('account_number', 'N/A')}\n"
                    f"Account Type: {user_info.get('account_type', 'N/A')}\n"
                    f"Account ID: {user_info.get('account_id', 'N/A')}")
         print(details + "\n")
         return {"task_result": details, "next_action": END, "error_message": None, "top_matches_from_l2": None}

    print(f"{supervisor_name} Final Routing Decision: {next_action}")
    return {
        "next_action": next_action,
        "top_matches_from_l2": final_top_matches, # Pass matches
        "error_message": error_message,
        "suggested_choices": None # Not decided here
        # Keep current_task as is
    }

# Define L2 Supervisor nodes using the helper
def l2_auth_supervisor(state: AppState) -> dict:
    return l2_supervisor_router(state, "L2_AuthSupervisor", l2_auth_tool_names, l2_auth_tool_keywords, l2_auth_tool_embeddings_np, fallback_node="L2_SupportSupervisor")

def l2_account_supervisor(state: AppState) -> dict:
    return l2_supervisor_router(state, "L2_AccountSupervisor", l2_account_tool_names, l2_account_tool_keywords, l2_account_tool_embeddings_np, fallback_node="L2_SupportSupervisor")

def l2_support_supervisor(state: AppState) -> dict:
    # Support supervisor fallback is Human Handoff if it can't route within its own tools
    return l2_supervisor_router(state, "L2_SupportSupervisor", l2_support_tool_names, l2_support_tool_keywords, l2_support_tool_embeddings_np, fallback_node="L3_HumanHandoffNode")


# -----------------------------------------------------------------------------
# --- Level 3 Tool Nodes (MODIFIED with Entry Check) ---
# -----------------------------------------------------------------------------

# --- Wrapper for L3 node ambiguity check ---
def l3_ambiguity_check(state: AppState, current_node_name: str) -> Optional[dict]:
    """Checks for ambiguity based on scores passed from L2.
       Returns state update dict for clarification if ambiguous, else None.
    """
    top_matches = state.get("top_matches_from_l2")

    # Proceed normally if no matches were passed (shouldn't happen often) or only one match
    if not top_matches or len(top_matches) < 2:
        # print(f"[{current_node_name}] No ambiguity check needed (matches: {top_matches})")
        return None # Not ambiguous or no data to check

    # Sanity check: Is this node *really* the top match?
    if top_matches[0][0] != current_node_name:
        print(f"Warning: [{current_node_name}] Mismatch! Routed here but wasn't top match ({top_matches[0][0]}). Proceeding cautiously.")
        # Decide how to handle: proceed, route to support, or ask clarification anyway?
        # Let's ask clarification based on the passed matches to be safe.
        pass # Continue to difference check below

    top_name, top_score = top_matches[0]
    second_name, second_score = top_matches[1]
    diff = top_score - second_score
    print(f"[{current_node_name}] Ambiguity Check: Top={top_score:.4f}, Second={second_score:.4f}, Diff={diff:.4f}")

    if diff <= DIFF_THRESHOLD:
        print(f"[{current_node_name}] Ambiguity detected (Diff <= {DIFF_THRESHOLD}). Routing to Clarification.")
        # Prepare choices (use top 2 or 3 from the passed list)
        choices_to_present = [
            (name, get_node_description(name)) for name, score in top_matches[:2] # Present top 2
        ]
        # Return state update to trigger clarification
        return {
            "next_action": "AskForClarificationNode",
            "suggested_choices": choices_to_present,
            "top_matches_from_l2": None, # Clear matches after using them
            "error_message": None # Not an error state
        }
    else:
        print(f"[{current_node_name}] High confidence execution (Diff > {DIFF_THRESHOLD}).")
        return None # Not ambiguous, proceed with tool execution

# --- Modified L3 Nodes ---
# Example modification for login_tool_node:
def login_tool_node(state: AppState) -> dict:
    """Node to handle the login process, checks ambiguity first."""
    # --- ENTRY AMBIGUITY CHECK ---
    ambiguity_result = l3_ambiguity_check(state, "L3_LoginToolNode")
    if ambiguity_result:
        return ambiguity_result # Reroute to clarification
    # --- END CHECK ---

    # --- ORIGINAL TOOL LOGIC ---
    print("--- Executing Login Tool (Confident) ---")
    email = input("Enter your email: ")
    password = input("Enter your password: ")
    user_data = local_db.get(email)
    result_data = {}
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

    # --- Add clearing of L2 matches to the return ---
    result_data["top_matches_from_l2"] = None
    return result_data

# Apply the same pattern to ALL other L3 nodes:
def signup_tool_node(state: AppState) -> dict:
    ambiguity_result = l3_ambiguity_check(state, "L3_SignupToolNode")
    if ambiguity_result: return ambiguity_result
    print("--- Executing Signup Tool (Confident) ---")
    # ... original signup logic ...
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
    account_number = generate_account_number()
    account_id = generate_account_id()
    account_type = "Savings" # Default
    local_db[email] = {
        "name": login_name.strip(), "account_holder_name": account_holder_name.strip(), "password": password,
        "balance": 0, "history": [], "account_number": account_number, "account_id": account_id, "account_type": account_type }
    result = f"Sign up successful, {login_name}! Your new {account_type} account ({account_number}) for {account_holder_name} is ready. You can now log in."
    print(result + "\n")
    return {"task_result": result, "error_message": None, "top_matches_from_l2": None}

def password_reset_tool_node(state: AppState) -> dict:
    ambiguity_result = l3_ambiguity_check(state, "L3_PasswordResetToolNode")
    if ambiguity_result: return ambiguity_result
    print("--- Executing Password Reset Tool (Confident) ---")
    # ... original password reset logic ...
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

def logout_tool_node(state: AppState) -> dict:
    ambiguity_result = l3_ambiguity_check(state, "L3_LogoutToolNode")
    if ambiguity_result: return ambiguity_result
    print("--- Executing Logout Tool (Confident) ---")
    # ... original logout logic ...
    if not state.get("user_info"):
        result = "You are not currently logged in."
        print(result + "\n")
        return {"task_result": result, "error_message": None, "top_matches_from_l2": None}
    else:
        logged_in_name = state["user_info"].get("name", "User")
        result = f"Logging out {logged_in_name}. You have been logged out successfully."
        print(result + "\n")
        return {"user_info": None, "task_result": result, "error_message": None, "top_matches_from_l2": None}

def check_balance_tool_node(state: AppState) -> dict:
    ambiguity_result = l3_ambiguity_check(state, "L3_CheckBalanceToolNode")
    if ambiguity_result: return ambiguity_result
    print("--- Executing Check Balance Tool (Confident) ---")
    # ... original check balance logic ...
    if not state.get("user_info"):
        result = "Error: You must be logged in to check your balance."
        print(result + "\n")
        return {"task_result": result, "error_message": "Authentication Required", "top_matches_from_l2": None}
    email = state["user_info"]["email"]
    balance = local_db[email].get("balance", None)
    if balance is None:
         result = "Error: Could not retrieve balance information."
         print(result + "\n")
         return {"task_result": result, "error_message": "Balance Data Missing", "top_matches_from_l2": None}
    account_number = state["user_info"].get("account_number", "N/A")
    result = f"Your current balance for account {account_number} is: ${balance:.2f}"
    print(result + "\n")
    return {"task_result": result, "error_message": None, "top_matches_from_l2": None}

def get_history_tool_node(state: AppState) -> dict:
    ambiguity_result = l3_ambiguity_check(state, "L3_GetHistoryToolNode")
    if ambiguity_result: return ambiguity_result
    print("--- Executing Get History Tool (Confident) ---")
    # ... original get history logic ...
    if not state.get("user_info"):
        result = "Error: You must be logged in to view transaction history."
        print(result + "\n")
        return {"task_result": result, "error_message": "Authentication Required", "top_matches_from_l2": None}
    email = state["user_info"]["email"]
    history = local_db[email].get("history", [])
    account_number = state["user_info"].get("account_number", "N/A")
    if history:
        history_str = "\n".join([f"- {item}" for item in history])
        result = f"Your recent transactions for account {account_number}:\n{history_str}"
    else:
        result = f"No transaction history found for account {account_number}."
    print(result + "\n")
    return {"task_result": result, "error_message": None, "top_matches_from_l2": None}

def login_name_update_tool_node(state: AppState) -> dict:
    ambiguity_result = l3_ambiguity_check(state, "L3_LoginNameUpdateToolNode")
    if ambiguity_result: return ambiguity_result
    print("--- Executing Login/Display Name Update Tool (Confident) ---")
    # ... original logic name update logic ...
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
    if email in local_db:
        local_db[email]["name"] = new_login_name
        updated_user_info = user_info.copy()
        updated_user_info["name"] = new_login_name
        result = f"Your login/display name has been updated to '{new_login_name}'."
        print(result + "\n")
        return {"task_result": result, "user_info": updated_user_info, "error_message": None, "top_matches_from_l2": None}
    else:
        result = "Error: Could not find your account details to update login name."
        print(result + "\n")
        return {"task_result": result, "error_message": "Account Data Mismatch", "user_info": None, "top_matches_from_l2": None}

def account_holder_name_update_tool_node(state: AppState) -> dict:
    ambiguity_result = l3_ambiguity_check(state, "L3_AccountHolderNameUpdateToolNode")
    if ambiguity_result: return ambiguity_result
    print("--- Executing Account Holder Name Update Tool (Confident) ---")
    # ... original account holder name update logic ...
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
    if email in local_db:
        local_db[email]["account_holder_name"] = new_holder_name
        updated_user_info = user_info.copy()
        updated_user_info["account_holder_name"] = new_holder_name
        result = f"The account holder name has been updated to '{new_holder_name}'."
        print(result + "\n")
        return {"task_result": result, "user_info": updated_user_info, "error_message": None, "top_matches_from_l2": None}
    else:
        result = "Error: Could not find your account details to update account holder name."
        print(result + "\n")
        return {"task_result": result, "error_message": "Account Data Mismatch", "user_info": None, "top_matches_from_l2": None}


def faq_tool_node(state: AppState) -> dict:
    ambiguity_result = l3_ambiguity_check(state, "L3_FAQToolNode")
    if ambiguity_result: return ambiguity_result
    print("--- Executing FAQ Tool (Confident) ---")
    # ... original FAQ logic ...
    last_user_message = ""
    task = state.get("current_task", "").lower()
    if not any(kw in task for kw in ["hour", "open", "contact", "phone", "location", "address"]):
         for msg in reversed(state['messages']):
             if isinstance(msg, HumanMessage):
                 last_user_message = msg.content.lower()
                 break
    else:
        last_user_message = task

    result_data = {}
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
        result_data = {"task_result": "I couldn't find a direct answer in the FAQ.", "error_message": "FAQ Not Found"}

    if "task_result" in result_data: print(f"FAQ Result: {result_data['task_result']}\n")
    result_data["top_matches_from_l2"] = None
    return result_data


def human_handoff_node(state: AppState) -> dict:
    # No ambiguity check needed for explicit handoff
    print("--- Executing Human Handoff ---")
    result = "Connecting you to a human agent..."
    time.sleep(1)
    print(result + "\n")
    # Clear matches state on handoff
    return {"task_result": result, "next_action": END, "error_message": None, "top_matches_from_l2": None, "suggested_choices": None}


# -----------------------------------------------------------------------------
# Graph Definition (MODIFIED EDGES)
# -----------------------------------------------------------------------------

# --- Node Name Constants ---
L1_SUPERVISOR = "L1_Supervisor"
L2_AUTH = "L2_AuthSupervisor"
L2_ACCOUNT = "L2_AccountSupervisor"
L2_SUPPORT = "L2_SupportSupervisor"
CLARIFICATION_NODE = "AskForClarificationNode"
L3_LOGIN = "L3_LoginToolNode"
L3_SIGNUP = "L3_SignupToolNode"
L3_PWRESET = "L3_PasswordResetToolNode"
L3_LOGOUT = "L3_LogoutToolNode"
L3_BALANCE = "L3_CheckBalanceToolNode"
L3_HISTORY = "L3_GetHistoryToolNode"
L3_LOGIN_NAME = "L3_LoginNameUpdateToolNode"
L3_HOLDER_NAME = "L3_AccountHolderNameUpdateToolNode"
L3_FAQ = "L3_FAQToolNode"
L3_HANDOFF = "L3_HumanHandoffNode"

ALL_L2_SUPERVISORS = [L2_AUTH, L2_ACCOUNT, L2_SUPPORT]
ALL_L3_TOOLS = [ L3_LOGIN, L3_SIGNUP, L3_PWRESET, L3_LOGOUT, L3_BALANCE, L3_HISTORY, L3_LOGIN_NAME, L3_HOLDER_NAME, L3_FAQ, L3_HANDOFF ]

# --- Routing Functions ---

def route_l1_decision(state: AppState) -> str:
    """Routes from L1. Expects next_action to be an L2 supervisor."""
    next_node = state.get("next_action")
    print(f"[Router] L1 Decision: Route to {next_node}")
    # L1 routes directly to L2 supervisors (or Auth if login needed)
    if next_node in ALL_L2_SUPERVISORS:
        return next_node
    else: # Fallback / Error case
        print(f"[Router] L1 Warning/Error: Invalid next_action '{next_node}' from L1. Defaulting to Support.")
        # Don't overwrite state here, let L2 Support handle if it gets there
        return L2_SUPPORT

def route_l2_decision(state: AppState) -> str:
    """Routes from L2. Expects next_action to be the top L3 tool or a fallback L2/L3."""
    next_node = state.get("next_action")
    print(f"[Router] L2 Decision: Route to {next_node}")
    # L2 should primarily route to L3 tools, or fallback to Support/Handoff
    if next_node in ALL_L3_TOOLS or next_node == L2_SUPPORT or next_node == L3_HANDOFF or next_node == L2_AUTH or next_node == END:
         # Allow L2_ACCOUNT to directly END for display details
        return next_node
    else: # Fallback / Error case
        print(f"[Router] L2 Warning/Error: Invalid next_action '{next_node}' from L2. Defaulting to Support.")
        # Don't overwrite state here
        return L2_SUPPORT

# --- NEW Router for L3 Nodes ---
def route_after_l3_decision(state: AppState) -> str:
    """Routes after an L3 node either executes or decides to clarify."""
    next_node_action = state.get("next_action") # Check if L3 set it to Clarification
    error_message = state.get("error_message")

    # 1. Check if L3 decided clarification is needed
    if next_node_action == CLARIFICATION_NODE:
        print(f"[Router] L3 decided Clarification. Routing to {CLARIFICATION_NODE}.")
        return CLARIFICATION_NODE

    # 2. If L3 executed, check for errors (mimic route_after_tool logic)
    print(f"[Router] L3 executed. Checking result/error.")
    if error_message:
        print(f"Error detected after L3 tool: {error_message}.")
        if error_message in ["Authentication Required", "Authentication Failed", "Account Data Mismatch"]:
             print("Routing to L2 Auth Supervisor.")
             # State should already be updated by L3 node, error message is present
             return L2_AUTH # L2 Auth will handle login flow
        elif error_message == "FAQ Not Found":
             print("Routing to L2 Support Supervisor.")
             return L2_SUPPORT # L2 Support handles FAQ not found -> Handoff
        else:
             print("Unhandled tool error. Ending turn.")
             return END # End turn, main loop shows error
    else:
        # Successful tool execution
        print("L3 Tool executed successfully. Ending turn.")
        return END


def route_after_clarification(state: AppState) -> str:
    """Routes from Clarification node based on user choice."""
    # --- THIS FUNCTION REMAINS UNCHANGED ---
    next_node = state.get("next_action")
    print(f"[Router] Clarification Decision: Route to {next_node}")
    # Validate the chosen node is a valid target (L3 or L2)
    if next_node in ALL_L3_TOOLS or next_node in ALL_L2_SUPERVISORS:
        return next_node
    else:
        print(f"[Router] Clarification Warning/Error: Invalid next_action '{next_node}'. Defaulting to Support.")
        # state["error_message"] = state.get("error_message") or f"Invalid Clarification Routing: {next_node}" # Maybe set error
        return L2_SUPPORT


# --- Build the graph ---
builder = StateGraph(AppState)

# Add ALL Nodes (including Clarification)
builder.add_node(L1_SUPERVISOR, l1_main_supervisor_rag)
builder.add_node(L2_AUTH, l2_auth_supervisor)
builder.add_node(L2_ACCOUNT, l2_account_supervisor)
builder.add_node(L2_SUPPORT, l2_support_supervisor)
builder.add_node(CLARIFICATION_NODE, ask_for_clarification_node) # NEW
builder.add_node(L3_LOGIN, login_tool_node)
builder.add_node(L3_SIGNUP, signup_tool_node)
builder.add_node(L3_PWRESET, password_reset_tool_node)
builder.add_node(L3_LOGOUT, logout_tool_node)
builder.add_node(L3_BALANCE, check_balance_tool_node)
builder.add_node(L3_HISTORY, get_history_tool_node)
builder.add_node(L3_LOGIN_NAME, login_name_update_tool_node)
builder.add_node(L3_HOLDER_NAME, account_holder_name_update_tool_node)
builder.add_node(L3_FAQ, faq_tool_node)
builder.add_node(L3_HANDOFF, human_handoff_node)


# --- Define Edges ---
builder.add_edge(START, L1_SUPERVISOR)

# L1 to L2 routing (No Clarification from L1)
builder.add_conditional_edges(
    L1_SUPERVISOR,
    route_l1_decision,
    {L2_AUTH: L2_AUTH, L2_ACCOUNT: L2_ACCOUNT, L2_SUPPORT: L2_SUPPORT} # L1 routes to L2
)

# L2 to L3/L2/END routing (No Clarification from L2)
l2_targets = {tool: tool for tool in ALL_L3_TOOLS}
l2_targets[L2_AUTH] = L2_AUTH # Allow Account -> Auth redirect
l2_targets[L2_SUPPORT] = L2_SUPPORT # Allow Auth/Account -> Support fallback
l2_targets[L3_HANDOFF] = L3_HANDOFF # Allow Support -> Handoff fallback
l2_targets[END] = END # Allow Account -> END
for supervisor_node in ALL_L2_SUPERVISORS:
    builder.add_conditional_edges(supervisor_node, route_l2_decision, l2_targets)

# --- NEW Edges FROM L3 Nodes ---
l3_targets = {
    CLARIFICATION_NODE: CLARIFICATION_NODE, # If L3 decides ambiguity
    END: END, # If L3 executes successfully
    L2_AUTH: L2_AUTH, # If L3 returns auth error
    L2_SUPPORT: L2_SUPPORT # If L3 returns FAQ error
}
for tool_node in ALL_L3_TOOLS:
    # Handoff node always goes to END, no ambiguity check needed
    if tool_node != L3_HANDOFF:
        builder.add_conditional_edges(tool_node, route_after_l3_decision, l3_targets)
    else:
         # Explicitly add edge for handoff to END (as it doesn't use the conditional router)
         builder.add_edge(L3_HANDOFF, END)


# Edges FROM Clarification Node
clarification_targets = {tool: tool for tool in ALL_L3_TOOLS} # User chooses an L3 tool
clarification_targets[L2_SUPPORT] = L2_SUPPORT # Fallback on clarification error
# Potentially allow clarification back to L2 supervisors? Less common.
# clarification_targets[L2_AUTH] = L2_AUTH
# clarification_targets[L2_ACCOUNT] = L2_ACCOUNT
builder.add_conditional_edges(CLARIFICATION_NODE, route_after_clarification, clarification_targets)


# Compile the graph
try:
    graph = builder.compile()
    print("\nGraph compiled successfully (Hybrid RAG + L3 Clarification Logic)!")
except Exception as e:
    print(f"\nFatal Error: Graph compilation failed: {e}")
    traceback.print_exc()
    exit(1)

# -----------------------------------------------------------------------------
# Main conversation loop (Unchanged)
# -----------------------------------------------------------------------------
def main():
    print("\n=== Welcome to the Multi-Level Banking Assistant (v8 - L3 Clarification) ===")
    # ... (rest of main loop is identical to previous version) ...
    print("You can ask about balance, history, FAQs, login, signup, password reset, logout, name updates.")
    print("Type 'quit' or 'exit' to end the conversation.")

    current_state: AppState = {
        "messages": [],
        "user_info": None,
        "current_task": None,
        "task_result": None,
        "next_action": None,
        "error_message": None,
        "suggested_choices": None,
        "top_matches_from_l2": None, # Initialize new field
    }

    while True:
        # Check if we need to print something *before* asking for user input (e.g., clarification result)
        final_task_result_prev_turn = current_state.get("task_result")
        if final_task_result_prev_turn:
            print(f"\nAssistant: {final_task_result_prev_turn}")
            # current_state['task_result'] = None # Clear after printing - handled in state reset below

        final_error_message_prev_turn = current_state.get("error_message")
        if final_error_message_prev_turn:
             if "Ambiguous request" in final_error_message_prev_turn:
                  # This should be less common now as clarification happens first
                  print(f"\nAssistant: I'm still not sure what you meant. {final_error_message_prev_turn}. Could you please ask for 'help'?")
             elif final_error_message_prev_turn == "Authentication Required":
                  print(f"\nAssistant: Please log in first to complete your request.")
             elif final_error_message_prev_turn in ["Authentication Failed", "Account Not Found", "Email Exists"]:
                  print(f"\nAssistant: There was an issue with authentication: {final_error_message_prev_turn}. Please try again.")
             elif "Invalid" in final_error_message_prev_turn or "Missing" in final_error_message_prev_turn:
                  print(f"\nAssistant: There was an input error: {final_error_message_prev_turn}. Please try again.")
             elif "Internal error" in final_error_message_prev_turn or "Clarification Routing" in final_error_message_prev_turn:
                  print(f"\nAssistant: Sorry, an internal routing error occurred. Please try again or ask for 'help'.")
             else: # Generic fallback error message
                  print(f"\nAssistant: Sorry, I encountered an issue: {final_error_message_prev_turn}. Please try asking differently or ask for 'help'.")
             # current_state['error_message'] = None # Clear after printing - handled in state reset below

        # --- Get User Input ---
        login_display_name = current_state.get("user_info", {}).get("name") if current_state.get("user_info") else None
        auth_display = f"(Logged in as: {login_display_name})" if login_display_name else "(Not Logged In)"
        user_input = input(f"\nYou {auth_display}: ")

        if 'quit' in user_input.lower() or 'exit' in user_input.lower():
            print("Banking Assistant: Goodbye!")
            break

        # --- Prepare state for the new turn ---
        current_messages = current_state.get('messages', [])
        if not isinstance(current_messages, list):
             current_messages = list(current_messages)
        current_messages.append(HumanMessage(content=user_input))

        # Reset turn-specific state, keep user_info and messages
        current_state = {
            "messages": current_messages,
            "user_info": current_state.get("user_info"),
            "current_task": None, # Will be set by L1
            "task_result": None,
            "next_action": None,
            "error_message": None,
            "suggested_choices": None,
            "top_matches_from_l2": None,
        }


        print("\nAssistant Processing...")
        try:
            final_state = None
            for event in graph.stream(current_state, {"recursion_limit": 25}):
                node_name = list(event.keys())[0]
                # Print supervisor/tool outputs, hide RAG util details
                if "_node" in node_name or "Supervisor" in node_name:
                     print(f"--- Event: Node '{node_name}' Output: {event[node_name]} ---")
                final_state = event[node_name] # Get the state *after* the node ran

            # Update central state with the final state from the stream
            if final_state:
                current_state.update(final_state)
            else:
                print("Warning: Graph stream finished without providing a final state update.")

            # Results/Errors/Completion messages handled at the start of the next loop

        except Exception as e:
             # ... (critical error handling) ...
            print(f"\n--- Critical Error during graph execution ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            traceback.print_exc()
            print("\nAssistant: I've encountered a critical system error. Please restart the conversation or try again later.")
            break

if __name__ == "__main__":
    main()