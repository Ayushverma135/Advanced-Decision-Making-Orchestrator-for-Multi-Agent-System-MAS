# --- Imports ---
import random
import string
import uuid
from typing import TypedDict, List, Optional, Sequence, Annotated, Tuple, Dict, Any # Added Dict, Any
import torch
import traceback
import re
import time
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# --- Import definitions and tools ---
from MAS6_hybrid.descriptions import (
    l1_route_definitions, l2_auth_tool_definitions, l2_account_tool_definitions,
    l2_support_tool_definitions, DISPLAY_DETAILS_ACTION, get_node_description
)
from MAS6_hybrid.tools import (
    login_tool_node, signup_tool_node, password_reset_tool_node, logout_tool_node,
    check_balance_tool_node, get_history_tool_node, login_name_update_tool_node,
    account_holder_name_update_tool_node, faq_tool_node, human_handoff_node
)

# -----------------------------------------------------------------------------
# Define DBs Here (or import from database.py if you create one)
# -----------------------------------------------------------------------------
# Local in-memory database
local_db = {
    "ayush@gmail.com": {
        "name": "ayush135", "account_holder_name": "Ayush Sharma", "password": "123",
        "balance": 1500.75, "history": ["+ $1000 (Initial Deposit)", "- $50 (Groceries)", "+ $600.75 (Salary)"],
        "account_number": ''.join(random.choices(string.digits, k=10)), # Use local helpers if needed
        "account_id": str(uuid.uuid4()), "account_type": "Savings"
    }
}
# Simple FAQ database
faq_db = {
    "hours": "Our bank branches are open Mon-Fri 9 AM to 5 PM. Online banking is available 24/7.",
    "contact": "You can call us at 1-800-BANKING or visit our website's contact page.",
    "locations": "We have branches in Pune and Gurugram. Use our online locator for specific addresses."
}

# -----------------------------------------------------------------------------
# State Definition (Needs to be defined here or imported)
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
    suggested_choices: Optional[List[Tuple[str, str]]]
    top_matches_from_l2: Optional[List[Tuple[str, float]]]

# -----------------------------------------------------------------------------
# --- Hybrid RAG Setup (Uses imported definitions) ---
# -----------------------------------------------------------------------------
HYBRID_ALPHA = 0.8
DIFF_THRESHOLD = 0.2 # <<< Ambiguity threshold
MIN_SCORE_THRESHOLD = 0.04

# Initialize Embedding Model
hf_model_name = "sentence-transformers/all-MiniLM-L6-v2"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = None
try:
    print(f"Using device: {device}")
    print(f"Loading HuggingFaceEmbeddings model: {hf_model_name}")
    embedding_model = HuggingFaceEmbeddings(model_name=hf_model_name, model_kwargs={'device': device})
    _ = embedding_model.embed_query("test initialization")
    print("HuggingFaceEmbeddings model loaded successfully.")
except Exception as e:
    print(f"\n--- Fatal Error Initializing Embedding Model ---")
    print(f"Error: {e}")
    traceback.print_exc()
    exit(1)

# Compute Embeddings (Uses imported definitions)
def compute_embeddings(definitions: List[dict]) -> tuple[List[str], List[List[str]], np.ndarray]:
    names = [d["name"] for d in definitions]
    keywords = [[kw.lower() for kw in d.get("keywords", [])] for d in definitions]
    descriptions = [d["description"] for d in definitions] # Need descriptions for embedding
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

# Keyword Scoring Function
def calculate_keyword_score(query: str, keywords: List[str]) -> float:
    if not keywords: return 0.0
    query_lower = query.lower()
    match_count = sum(1 for kw in keywords if kw in query_lower)
    score = float(match_count) / float(len(keywords)) if keywords else 0.0
    return min(score, 1.0)

# Hybrid RAG Utility Function
def get_top_k_hybrid_matches(
    query: str, names: List[str], keywords_list: List[List[str]], embeddings: np.ndarray,
    k: int = 2, alpha: float = HYBRID_ALPHA, min_score_threshold: float = MIN_SCORE_THRESHOLD
) -> List[Tuple[str, float]]:
    # ... (Implementation remains the same) ...
    if not query or not query.strip(): print("[Hybrid RAG Util] Warning: Empty query provided."); return []
    num_options = len(names)
    if num_options == 0 or len(keywords_list) != num_options or embeddings.shape[0] != num_options: print("[Hybrid RAG Util] Error: Mismatched input list lengths."); return []
    results = []
    try:
        query_embedding_list = embedding_model.embed_query(query)
        query_embedding_np = np.array(query_embedding_list).reshape(1, -1)
        if embeddings.ndim == 1: embeddings = embeddings.reshape(1, -1)
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

# --- L3 Ambiguity Check Helper (Needs to be defined in main or imported by tools.py) ---
def l3_ambiguity_check(state: AppState, current_node_name: str) -> Optional[Dict[str, Any]]:
    """Checks for ambiguity based on scores passed from L2.
       Returns state update dict for clarification if ambiguous, else None.
    """
    top_matches = state.get("top_matches_from_l2")
    if not top_matches or len(top_matches) < 2: return None

    if top_matches[0][0] != current_node_name:
        print(f"Warning: [{current_node_name}] Mismatch! Routed here but wasn't top match ({top_matches[0][0]}). Checking ambiguity anyway.")

    top_name, top_score = top_matches[0]
    second_name, second_score = top_matches[1]
    diff = top_score - second_score
    print(f"[{current_node_name}] Ambiguity Check: Top={top_score:.4f}, Second={second_score:.4f}, Diff={diff:.4f}")

    if diff <= DIFF_THRESHOLD:
        print(f"[{current_node_name}] Ambiguity detected (Diff <= {DIFF_THRESHOLD}). Routing to Clarification.")
        choices_to_present = [ (name, get_node_description(name)) for name, score in top_matches[:2] ]
        return {
            "next_action": "AskForClarificationNode",
            "suggested_choices": choices_to_present,
            "top_matches_from_l2": None,
            "error_message": None
        }
    else:
        print(f"[{current_node_name}] High confidence execution (Diff > {DIFF_THRESHOLD}).")
        return None


# --- Clarification Node (Needs to be defined here or imported by main) ---
def ask_for_clarification_node(state: AppState) -> dict:
    # ... (Implementation remains the same) ...
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
# --- Level 1 & 2 Supervisors (Unchanged logic - pass results down) ---
# -----------------------------------------------------------------------------
# ... (l1_main_supervisor_rag, l2_supervisor_router, l2_auth_supervisor, l2_account_supervisor, l2_support_supervisor functions remain the same as previous version v8_l3_clarify) ...
# -----------------------------------------------------------------------------
# --- Level 1 Supervisor (Passes Results) ---
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
# --- Level 2 Supervisors (Pass Results) ---
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
    # Need to handle errors passed to it specifically before routing
    error_msg_in = state.get("error_message")
    specific_errors = [
        "Unknown Auth Task", "Unknown Account Task", "Unclear authentication request",
        "Unclear account request", "L1 Routing Failed", "Invalid L1 Routing Decision",
        "Invalid L2 Routing Decision", "L1 Embedding Error", "Could not understand request",
        "Error during clarification", "Internal error: Missing clarification choices"
        # Don't include "Ambiguous" errors here, let L2 Support try to route first
    ]
    if error_msg_in and any(err_pattern in error_msg_in for err_pattern in specific_errors):
        print(f"L2 Support: Handling specific error '{error_msg_in}', forcing handoff.")
        return {"next_action": "L3_HumanHandoffNode", "error_message": None, "top_matches_from_l2": None}

    return l2_supervisor_router(state, "L2_SupportSupervisor", l2_support_tool_names, l2_support_tool_keywords, l2_support_tool_embeddings_np, fallback_node="L3_HumanHandoffNode")


# --- Imports (unchanged) ---
# --- Langchain/Sklearn Imports (unchanged) ---
# --- Helper functions, DBs, State (unchanged) ---
# --- L3 Tool Nodes (unchanged - including ambiguity check) ---
# --- RAG Setup (unchanged) ---
# --- Keyword/Hybrid/Ambiguity functions (unchanged) ---
# --- Clarification Node (unchanged) ---
# --- L1/L2 Supervisors (unchanged) ---

# ... (Keep all previous code sections up to Graph Definition) ...

# -----------------------------------------------------------------------------
# Graph Definition (MODIFIED Routing Logic)
# -----------------------------------------------------------------------------

# --- Node Name Constants (unchanged) ---
# ... (L1_SUPERVISOR, L2_AUTH, ..., L3_HANDOFF, ALL_L2_SUPERVISORS, ALL_L3_TOOLS) ...
L1_SUPERVISOR = "L1_Supervisor"
L2_AUTH = "L2_AuthSupervisor"
L2_ACCOUNT = "L2_AccountSupervisor"
L2_SUPPORT = "L2_SupportSupervisor"
CLARIFICATION_NODE = "AskForClarificationNode" # New node name

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


# --- Routing Functions (MODIFIED) ---

def route_l1_decision(state: AppState) -> str:
    # --- UNCHANGED ---
    next_node = state.get("next_action")
    print(f"[Router] L1 Decision: Route to {next_node}")
    if next_node in ALL_L2_SUPERVISORS: return next_node
    else: print(f"[Router] L1 Warning/Error: Invalid next_action '{next_node}' from L1. Defaulting to Support."); return L2_SUPPORT

def route_l2_decision(state: AppState) -> str:
    # --- UNCHANGED ---
    next_node = state.get("next_action")
    print(f"[Router] L2 Decision: Route to {next_node}")
    if next_node in ALL_L3_TOOLS or next_node == END or next_node in [L2_AUTH, L2_SUPPORT, L3_HANDOFF]: return next_node
    else: print(f"[Router] L2 Warning/Error: Invalid next_action '{next_node}' from L2. Defaulting to Support."); return L2_SUPPORT

# --- Router specifically AFTER L3 Nodes ---
def route_after_l3_tool(state: AppState) -> str:
    """Routes after an L3 node finishes execution OR decides to clarify."""
    next_node_action = state.get("next_action") # Check if L3 set it (e.g., to Clarification)
    error_message = state.get("error_message")

    # 1. Check if L3 decided clarification is needed
    if next_node_action == CLARIFICATION_NODE:
        print(f"[Router] After L3 Tool: Routing to {CLARIFICATION_NODE}.")
        return CLARIFICATION_NODE # Go ask the user

    # 2. If L3 executed (next_action is None or END), check for errors
    print(f"[Router] After L3 Tool Execution. Checking result/error. next_action='{next_node_action}'")
    if error_message:
        print(f"Error detected after L3 tool: {error_message}.")
        if error_message in ["Authentication Required", "Authentication Failed", "Account Data Mismatch"]:
             print("Routing to L2 Auth Supervisor.")
             return L2_AUTH
        elif error_message == "FAQ Not Found":
             print("Routing to L2 Support Supervisor.")
             return L2_SUPPORT
        else:
             print("Unhandled tool error. Ending turn.")
             return END
    else:
        # Successful tool execution or forced END (like Handoff)
        print("L3 Tool successful or END signal received. Ending turn.")
        return END

# --- Router specifically AFTER Clarification Node ---
def route_after_clarification(state: AppState) -> str:
    """Routes from Clarification node based on user choice."""
    # --- THIS FUNCTION REMAINS MOSTLY UNCHANGED ---
    # --- but we simplify the target validation slightly ---
    next_node = state.get("next_action") # Should be set to a chosen L3 node
    print(f"[Router] After Clarification: Routing to {next_node}")
    # Validate the chosen node is a valid L3 tool (or fallback to Support)
    if next_node in ALL_L3_TOOLS:
        return next_node
    else:
        print(f"[Router] Clarification Warning/Error: Invalid next_action '{next_node}'. Defaulting to Support.")
        return L2_SUPPORT


# --- Build the graph ---
builder = StateGraph(AppState)

# Add ALL Nodes (unchanged)
# ... (builder.add_node calls for all nodes) ...
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

# L1 -> L2 (Unchanged)
builder.add_conditional_edges( L1_SUPERVISOR, route_l1_decision,
    {L2_AUTH: L2_AUTH, L2_ACCOUNT: L2_ACCOUNT, L2_SUPPORT: L2_SUPPORT}
)

# L2 -> L3 (or fallback L2/L3/END) (Unchanged)
l2_targets = {tool: tool for tool in ALL_L3_TOOLS}
l2_targets[L2_AUTH] = L2_AUTH
l2_targets[L2_SUPPORT] = L2_SUPPORT
l2_targets[L3_HANDOFF] = L3_HANDOFF
l2_targets[END] = END
for supervisor_node in ALL_L2_SUPERVISORS:
    builder.add_conditional_edges(supervisor_node, route_l2_decision, l2_targets)

# --- MODIFIED Edges FROM L3 Nodes --- using route_after_l3_tool
l3_targets = {
    CLARIFICATION_NODE: CLARIFICATION_NODE, # If L3 triggers clarification
    END: END,                               # If L3 succeeds or has unhandled error
    L2_AUTH: L2_AUTH,                       # If L3 returns auth error
    L2_SUPPORT: L2_SUPPORT                  # If L3 returns FAQ error
}
for tool_node in ALL_L3_TOOLS:
    builder.add_conditional_edges(tool_node, route_after_l3_tool, l3_targets)
    # Note: The L3_Handoff node returns next_action=END, so route_after_l3_tool
    # will correctly return END for it, directing it to the graph's END state.

# --- Edges FROM Clarification Node --- using route_after_clarification
clarification_targets = {tool: tool for tool in ALL_L3_TOOLS} # User chooses L3
clarification_targets[L2_SUPPORT] = L2_SUPPORT # Fallback
builder.add_conditional_edges(
    CLARIFICATION_NODE,
    route_after_clarification, # This router expects next_action to be an L3 tool
    clarification_targets
)

# Compile the graph
try:
    graph = builder.compile()
    print("\nGraph compiled successfully (Hybrid RAG + L3 Clarification - v3 Fix)!")
except Exception as e:
    print(f"\nFatal Error: Graph compilation failed: {e}")
    traceback.print_exc()
    exit(1)

# -----------------------------------------------------------------------------
# Main conversation loop (Unchanged)
# -----------------------------------------------------------------------------
# ... (main function remains the same) ...
def main():
    print("\n=== Welcome to the Multi-Level Banking Assistant (v9 - L3 Clarification Fix) ===")
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
        # Print results/errors from *previous* turn before asking for input
        final_task_result_prev_turn = current_state.get("task_result")
        if final_task_result_prev_turn:
            print(f"\nAssistant: {final_task_result_prev_turn}")

        final_error_message_prev_turn = current_state.get("error_message")
        if final_error_message_prev_turn:
             # ... (error message formatting - unchanged) ...
             if "Ambiguous request" in final_error_message_prev_turn:
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
            "top_matches_from_l2": None, # Clear previous matches
        }


        print("\nAssistant Processing...")
        try:
            final_state_update = None
            for event in graph.stream(current_state, {"recursion_limit": 25}):
                node_name = list(event.keys())[0]
                if "_node" in node_name or "Supervisor" in node_name:
                     print(f"--- Event: Node '{node_name}' Output: {event[node_name]} ---")
                final_state_update = event[node_name]

            if final_state_update:
                 current_state.update(final_state_update)
            else:
                print("Warning: Graph stream finished without providing a final state update.")

            # Post-turn check handled at start of next loop

        except Exception as e:
             # ... (critical error handling) ...
            print(f"\n--- Critical Error during graph execution ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            traceback.print_exc()
            print("\nAssistant: I've encountered a critical system error. Please restart the conversation or try again later.")
            break # Stop processing on critical errors

if __name__ == "__main__":
    main()