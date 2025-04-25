# --- Imports ---
import random
import string
import uuid
from typing import TypedDict, List, Optional, Sequence, Annotated, Tuple, Dict, Any
import torch
import traceback
import re
import time
import json # <<< For parsing LLM output
from LLMaas import LLMaaSModel
# --- LLM Setup (Replace with your actual LLM loading) ---
# Assuming LLMaaSModel was the intended way
try:
    # from LLMaas import LLMaaSModel # Your original import
    llmaas_model_instance = LLMaaSModel()
    llm = llmaas_model_instance.get_model()

    # Placeholder using a Langchain Chat model if LLMaaSModel isn't setup
    # Replace with your actual LLM integration
    # from langchain_openai import ChatOpenAI # Example
    # from langchain_groq import ChatGroq # Example
    # from langchain_anthropic import ChatAnthropic # Example
    import os
    # Make sure OPENAI_API_KEY (or equivalent for other models) is set as environment variable
    # os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0) # Use a cheap, fast model for routing

    print(f"Using LLM: {llm.model_name if hasattr(llm, 'model_name') else type(llm)}")
except ImportError as e:
     print(f"Error importing Langchain LLM: {e}. Please install required packages (e.g., pip install langchain-openai).")
     exit(1)
except Exception as e:
    print(f"Fatal Error: Could not initialize LLM model: {e}")
    exit(1)

# --- Other Imports ---
from langchain_huggingface import HuggingFaceEmbeddings # Keep for potential future use? Maybe remove if not needed
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# --- Import definitions and tools ---
from descriptions import (
    l1_route_definitions, l2_auth_tool_definitions, l2_account_tool_definitions,
    l2_support_tool_definitions, DISPLAY_DETAILS_ACTION, get_node_description
)
# Import tool functions from tools.py
from tools import (
    login_tool_node, signup_tool_node, password_reset_tool_node, logout_tool_node,
    check_balance_tool_node, get_history_tool_node, login_name_update_tool_node,
    account_holder_name_update_tool_node, faq_tool_node, human_handoff_node
)
# Import state and DBs from common_defs.py
from common_defs import AppState, UserInfo, local_db, faq_db

# -----------------------------------------------------------------------------
# --- LLM Prompts ---
# -----------------------------------------------------------------------------

# --- L1 Supervisor Prompt ---
L1_SYSTEM_PROMPT = """You are the main routing supervisor for a banking assistant.
Your goal is to determine the **single most relevant** department (L2 Supervisor) to handle the user's latest request, considering the conversation history and authentication status.

Available Departments (L2 Supervisors):
{l2_supervisor_descriptions}

Rules:
- Analyze the latest user message in the context of the conversation.
- Determine the primary intent.
- If the intent clearly matches an L2_AccountSupervisor task (balance, history, update names, view details) BUT the user is "Not Logged In", you MUST route to L2_AuthSupervisor first.
- Otherwise, choose the single best matching L2 Supervisor.
- Respond ONLY with the chosen supervisor's name (e.g., L2_AuthSupervisor, L2_AccountSupervisor, L2_SupportSupervisor). Do not add any other text or explanation."""

# --- L2 Supervisor Prompt Template ---
L2_SYSTEM_PROMPT_TEMPLATE = """You are the {supervisor_name}, responsible for routing user requests to the correct specialized tool within your department.
Your goal is to identify all relevant tools for the user's latest request and assign a relevance score (0.0 to 1.0).

Available Tools in your department:
{l3_tool_descriptions}

Rules:
- Analyze the user's request: "{user_request}"
- Identify ALL tools from the list above that could potentially handle the request.
- For each potentially relevant tool, assign a relevance score between 0.0 (not relevant) and 1.0 (highly relevant). A score >= 0.2 indicates relevance.
- **Crucially:** Respond ONLY with a valid JSON list of dictionaries. Each dictionary must have keys "name" (the tool's node name) and "score" (a float between 0.0 and 1.0).
- Only include tools with a relevance score >= 0.2 in the list.
- If no tools are relevant (score < 0.2), respond with an empty JSON list: []

Example Output (for a request like 'check my balance'):
[
  {{"name": "L3_CheckBalanceToolNode", "score": 0.95}},
  {{"name": "L3_GetHistoryToolNode", "score": 0.55}}
]

Example Output (if no tools match):
[]
"""

# Helper to format descriptions for prompts
def format_descriptions_for_prompt(definitions: List[Dict]) -> str:
    return "\n".join([f"- {d['name']}: {d['description']}" for d in definitions])

# Pre-format descriptions
L1_PROMPT_CONTEXT = format_descriptions_for_prompt(l1_route_definitions)
L2_AUTH_CONTEXT = format_descriptions_for_prompt(l2_auth_tool_definitions)
L2_ACCOUNT_CONTEXT = format_descriptions_for_prompt(l2_account_tool_definitions)
L2_SUPPORT_CONTEXT = format_descriptions_for_prompt(l2_support_tool_definitions)

# -----------------------------------------------------------------------------
# --- Clarification Node (from common_defs or defined here) ---
# -----------------------------------------------------------------------------
# Assuming it's defined in common_defs now, otherwise define ask_for_clarification_node here
from common_defs import ask_for_clarification_node # Make sure it's defined/imported

# -----------------------------------------------------------------------------
# --- Level 1 Supervisor (LLM Based) ---
# -----------------------------------------------------------------------------
def l1_main_supervisor_llm(state: AppState) -> dict:
    """Routes user requests to L2 supervisors using LLM."""
    print("\n--- L1 Main Supervisor (LLM Based) ---")
    is_logged_in = bool(state.get('user_info'))
    auth_status = "Logged In" if is_logged_in else "Not Logged In"
    print(f"Current Auth Status: {auth_status}")

    messages = state.get('messages', [])
    if not messages or not isinstance(messages[-1], HumanMessage):
        print("L1 LLM Warning: No user message found. Routing to Support.")
        return {"next_action": "L2_SupportSupervisor", "current_task": "Error: No user message", "error_message": "Missing User Input"}

    last_user_message = messages[-1].content
    print(f"User Message: '{last_user_message}'")

    # Prepare prompt
    system_prompt = L1_SYSTEM_PROMPT.format(l2_supervisor_descriptions=L1_PROMPT_CONTEXT)
    prompt_messages = [SystemMessage(content=system_prompt)]
    # Add recent history + auth status
    prompt_messages.append(SystemMessage(content=f"Current Authentication Status: {auth_status}"))
    prompt_messages.extend(messages[-5:]) # Include last 5 messages for context

    next_action = "L2_SupportSupervisor" # Default fallback
    error_message = None

    try:
        response = llm.invoke(prompt_messages)
        llm_decision = response.content.strip()
        print(f"L1 LLM Decision: {llm_decision}")

        # Basic validation of LLM output
        valid_l2_routes = [r["name"] for r in l1_route_definitions]
        if llm_decision in valid_l2_routes:
            # Apply login override rule (already instructed in prompt, but double-check)
            if llm_decision == "L2_AccountSupervisor" and not is_logged_in:
                print("L1 LLM Rule Applied: Account task requires login. Routing to Auth.")
                next_action = "L2_AuthSupervisor"
            else:
                next_action = llm_decision
        else:
            print(f"L1 LLM Warning: Invalid route '{llm_decision}' received. Defaulting to Support.")
            error_message = "L1 routing failed (invalid LLM response)"
            next_action = "L2_SupportSupervisor"

    except Exception as e:
        print(f"L1 LLM Error: Exception during LLM call: {e}")
        traceback.print_exc()
        error_message = f"L1 LLM Error: {e}"
        next_action = "L2_SupportSupervisor"

    print(f"L1 Final Routing Decision: {next_action}")
    return {
        "next_action": next_action,
        "current_task": last_user_message, # Pass original task
        "error_message": error_message,
        # Clear fields potentially set by previous turns/errors
        "suggested_choices": None,
    }

# -----------------------------------------------------------------------------
# --- Level 2 Supervisors (LLM Based) ---
# -----------------------------------------------------------------------------

# Helper function for L2 LLM routing logic
def l2_llm_router_logic(
    state: AppState,
    supervisor_name: str,
    tool_definitions: List[Dict],
    tool_context: str,
    fallback_node: str = "L3_HumanHandoffNode" # Default fallback for L2 failures
) -> dict:
    """Generic L2 logic using LLM to find relevant L3 tools."""
    print(f"\n--- {supervisor_name} (LLM Based) ---")
    task = state.get("current_task", "")
    print(f"Received Task: '{task}'")

    # --- Login Check (Specific to Account Supervisor) ---
    if supervisor_name == "L2_AccountSupervisor" and not state.get("user_info"):
        print("L2 Account: Authentication required. Routing to L2 Auth Supervisor.")
        # Important: Return immediately, don't call LLM
        return {
            "next_action": "L2_AuthSupervisor",
            "current_task": f"login (required for: {task or 'your request'})",
            "error_message": "Authentication Required"
        }
    # --- End Login Check ---

    # --- Pre-check for errors passed TO Support Supervisor ---
    # (Support supervisor needs to handle specific errors before calling LLM)
    if supervisor_name == "L2_SupportSupervisor":
        error_msg_in = state.get("error_message")
        specific_errors = [
             "FAQ Not Found", "Unknown Auth Task", "Unknown Account Task",
             "Unclear authentication request", "Unclear account request",
             "L1 Routing Failed", "Invalid L1 Routing Decision", "Invalid L2 Routing Decision",
             "L1 Embedding Error", "L1 LLM Error", "Could not understand request",
             "Error during clarification", "Internal error: Missing clarification choices"
             # No "Ambiguous" here, as L2 decides ambiguity now
        ]
        if error_msg_in and any(err_pattern in error_msg_in for err_pattern in specific_errors):
            print(f"L2 Support: Handling specific error '{error_msg_in}', forcing handoff.")
            return {"next_action": "L3_HumanHandoffNode", "error_message": None}
    # --- End Support Pre-check ---


    # --- Call LLM for L3 routing ---
    next_action = fallback_node
    error_message = None
    suggested_choices = None

    try:
        system_prompt = L2_SYSTEM_PROMPT_TEMPLATE.format(
            supervisor_name=supervisor_name,
            l3_tool_descriptions=tool_context,
            user_request=task
        )
        prompt_messages = [SystemMessage(content=system_prompt)]
        # Maybe add last user message again? Or keep context minimal?
        # prompt_messages.append(HumanMessage(content=task))

        response = llm.invoke(prompt_messages)
        llm_output_str = response.content.strip()
        print(f"{supervisor_name} LLM Decision Raw: {llm_output_str}")

        # --- Parse LLM JSON output ---
        relevant_tools = []
        try:
            # Handle potential markdown code fences ```json ... ```
            if llm_output_str.startswith("```json"):
                llm_output_str = llm_output_str[7:]
                if llm_output_str.endswith("```"):
                    llm_output_str = llm_output_str[:-3]
                llm_output_str = llm_output_str.strip()

            parsed_output = json.loads(llm_output_str)

            # Validate structure
            if isinstance(parsed_output, list):
                valid_tools = [t["name"] for t in tool_definitions]
                for item in parsed_output:
                    if (isinstance(item, dict) and
                        "name" in item and "score" in item and
                        item["name"] in valid_tools and
                        isinstance(item["score"], (int, float)) and
                        0.0 <= item["score"] <= 1.0):
                        # Filter based on score (redundant if prompt works, but safe)
                        if item["score"] >= 0.2:
                             relevant_tools.append(item)
                    else:
                        print(f"Warning: Invalid item format in LLM output: {item}")
                # Sort by score descending
                relevant_tools.sort(key=lambda x: x["score"], reverse=True)
            else:
                 raise ValueError("LLM output is not a JSON list.")

        except (json.JSONDecodeError, ValueError, TypeError) as json_e:
            print(f"{supervisor_name} LLM Error: Failed to parse JSON output: {json_e}")
            print(f"LLM Raw Output was: {llm_output_str}")
            error_message = f"{supervisor_name} LLM parsing error."
            # next_action remains fallback_node

        print(f"{supervisor_name} Relevant Tools Found: {relevant_tools}")

        # --- Apply Routing Logic based on number of relevant tools ---
        num_relevant = len(relevant_tools)
        if num_relevant == 0:
            print(f"{supervisor_name}: No relevant tools found by LLM. Routing to fallback {fallback_node}.")
            error_message = error_message or f"No suitable tool found in {supervisor_name}."
            next_action = fallback_node
        elif num_relevant == 1:
            tool_name = relevant_tools[0]["name"]
            print(f"{supervisor_name}: Exactly one relevant tool found ({tool_name}). Routing directly.")
            next_action = tool_name
            error_message = None # Clear any previous non-critical error
        elif num_relevant == 2:
            print(f"{supervisor_name}: Two relevant tools found. Routing to clarification.")
            next_action = "AskForClarificationNode"
            suggested_choices = [(t["name"], get_node_description(t["name"])) for t in relevant_tools]
            error_message = None
        else: # More than 2 relevant tools
            print(f"{supervisor_name}: More than two relevant tools found. Routing to clarification (Top 3).")
            next_action = "AskForClarificationNode"
            suggested_choices = [(t["name"], get_node_description(t["name"])) for t in relevant_tools[:3]] # Show top 3
            error_message = None

    except Exception as llm_e:
        print(f"{supervisor_name} LLM Error: Exception during LLM call: {llm_e}")
        traceback.print_exc()
        error_message = f"{supervisor_name} LLM Error: {llm_e}"
        next_action = fallback_node # Go to fallback on LLM error

    # Handle Account Supervisor's special display action case
    # Needs to happen AFTER LLM routing logic, only if direct routing is chosen
    if supervisor_name == "L2_AccountSupervisor" and next_action == DISPLAY_DETAILS_ACTION:
         print("L2 Account: Handling display details action directly.")
         user_info = state["user_info"]
         details = (f"Login/Display Name: {user_info.get('name', 'N/A')}\n"
                    f"Account Holder Name: {user_info.get('account_holder_name', 'N/A')}\n"
                    f"Account Number: {user_info.get('account_number', 'N/A')}\n"
                    f"Account Type: {user_info.get('account_type', 'N/A')}\n"
                    f"Account ID: {user_info.get('account_id', 'N/A')}")
         print(details + "\n")
         return {"task_result": details, "next_action": END, "error_message": None, "suggested_choices": None}

    print(f"{supervisor_name} Final Routing Decision: {next_action}")
    return {
        "next_action": next_action,
        "suggested_choices": suggested_choices,
        "error_message": error_message,
        # Don't need to pass top_matches_from_l2 anymore
    }

# Define L2 Supervisor nodes using the helper
def l2_auth_supervisor(state: AppState) -> dict:
    return l2_llm_router_logic(state, "L2_AuthSupervisor", l2_auth_tool_definitions, L2_AUTH_CONTEXT, fallback_node="L2_SupportSupervisor")

def l2_account_supervisor(state: AppState) -> dict:
    return l2_llm_router_logic(state, "L2_AccountSupervisor", l2_account_tool_definitions, L2_ACCOUNT_CONTEXT, fallback_node="L2_SupportSupervisor")

def l2_support_supervisor(state: AppState) -> dict:
    return l2_llm_router_logic(state, "L2_SupportSupervisor", l2_support_tool_definitions, L2_SUPPORT_CONTEXT, fallback_node="L3_HumanHandoffNode")


# -----------------------------------------------------------------------------
# Graph Definition (Adjusted Edges)
# -----------------------------------------------------------------------------

# --- Node Name Constants (unchanged) ---
# ... (all node name constants) ...
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

# --- Routing Functions ---

def route_l1_decision(state: AppState) -> str:
    """Routes from L1 (LLM based) to L2."""
    # --- UNCHANGED ---
    next_node = state.get("next_action")
    print(f"[Router] L1 Decision: Route to {next_node}")
    if next_node in ALL_L2_SUPERVISORS: return next_node
    else: print(f"[Router] L1 Warning/Error: Invalid next_action '{next_node}' from L1. Defaulting to Support."); return L2_SUPPORT

def route_l2_decision(state: AppState) -> str:
    """Routes from L2 (LLM based) to L3, Clarification, fallback L2/L3, or END."""
    # --- MODIFIED: Added CLARIFICATION_NODE as valid target ---
    next_node = state.get("next_action")
    print(f"[Router] L2 Decision: Route to {next_node}")
    valid_targets = ALL_L3_TOOLS + [END, L2_AUTH, L2_SUPPORT, L3_HANDOFF, CLARIFICATION_NODE]
    if next_node in valid_targets:
        return next_node
    else:
        print(f"[Router] L2 Warning/Error: Invalid next_action '{next_node}' from L2. Defaulting to Support.")
        return L2_SUPPORT

# --- Router AFTER L3 Tool (Simpler now) ---
def route_after_tool(state: AppState) -> str:
    """Routes after L3 tool execution based on success/error."""
    # --- THIS IS THE SIMPLER ROUTER USED BEFORE L3 CLARIFICATION ---
    print(f"[Router] After Tool Execution...")
    error_message = state.get("error_message")
    next_action_forced_by_tool = state.get("next_action")

    if next_action_forced_by_tool == END: # e.g. Handoff sets next_action=END
        print("Routing to END (forced by tool).")
        return END

    if error_message:
        print(f"Error detected after tool: {error_message}.")
        if error_message in ["Authentication Required", "Authentication Failed", "Account Data Mismatch"]:
             print("Authentication error detected. Routing to L2 Auth Supervisor.")
             # State should include cleared user_info if needed (done by tool)
             # Error message is present, L2 Auth can see it if needed
             return L2_AUTH
        elif error_message == "FAQ Not Found":
             print("FAQ not found error. Routing to L2 Support Supervisor for handoff.")
             return L2_SUPPORT
        else:
             print("Ending turn due to unhandled tool error.")
             return END # Main loop shows error
    else:
        print("Tool executed successfully. Ending current turn.")
        return END

# --- Router AFTER Clarification (Unchanged) ---
def route_after_clarification(state: AppState) -> str:
    """Routes from Clarification node based on user choice OR cancellation."""
    next_node = state.get("next_action") # Will be L3 tool name or None

    if next_node in ALL_L3_TOOLS: # User made a valid choice
        print(f"[Router] After Clarification: Routing to chosen L3 node: {next_node}")
        return next_node
    # --- ADDED CHECK FOR CANCELLATION ---
    elif next_node is None: # User likely cancelled via 'exit'
        print("[Router] After Clarification: No next action set (likely cancelled). Ending turn.")
        return END
    # --- END ADDED CHECK ---
    else: # Fallback for invalid next_action set somehow
        print(f"[Router] Clarification Warning/Error: Invalid next_action '{next_node}'. Defaulting to Support.")
        return L2_SUPPORT


# --- Build the graph ---
builder = StateGraph(AppState)

# Add ALL Nodes (unchanged)
# ... (builder.add_node calls) ...
builder.add_node(L1_SUPERVISOR, l1_main_supervisor_llm) # Use LLM L1
builder.add_node(L2_AUTH, l2_auth_supervisor)
builder.add_node(L2_ACCOUNT, l2_account_supervisor)
builder.add_node(L2_SUPPORT, l2_support_supervisor)
builder.add_node(CLARIFICATION_NODE, ask_for_clarification_node)
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

# L2 -> L3 / Clarification / L2 / END
l2_targets = {tool: tool for tool in ALL_L3_TOOLS}
l2_targets[CLARIFICATION_NODE] = CLARIFICATION_NODE # <<< L2 can now route here
l2_targets[L2_AUTH] = L2_AUTH
l2_targets[L2_SUPPORT] = L2_SUPPORT
l2_targets[L3_HANDOFF] = L3_HANDOFF
l2_targets[END] = END
for supervisor_node in ALL_L2_SUPERVISORS:
    builder.add_conditional_edges(supervisor_node, route_l2_decision, l2_targets)

# --- Edges FROM L3 Nodes --- using simple route_after_tool
# L3 nodes no longer decide ambiguity, they just finish or error out.
after_tool_map = { L2_AUTH: L2_AUTH, L2_SUPPORT: L2_SUPPORT, END: END }
for tool_node in ALL_L3_TOOLS:
     builder.add_conditional_edges(
         tool_node,
         route_after_tool, # Use the simpler router
         after_tool_map
     )

# --- Edges FROM Clarification Node --- using route_after_clarification (Unchanged)
clarification_targets = {tool: tool for tool in ALL_L3_TOOLS} # Expects L3 choice
clarification_targets[L2_SUPPORT] = L2_SUPPORT # Fallback
clarification_targets[END] = END # <<< ADDED: Allow routing to END on cancellation
# Could potentially add L2 supervisors if clarification node might suggest them
# clarification_targets[L2_AUTH] = L2_AUTH
# clarification_targets[L2_ACCOUNT] = L2_ACCOUNT
builder.add_conditional_edges(
    CLARIFICATION_NODE,
    route_after_clarification,
    clarification_targets
)

# Compile the graph
try:
    graph = builder.compile()
    print("\nGraph compiled successfully (LLM Routing + Clarification)!")
except Exception as e:
    print(f"\nFatal Error: Graph compilation failed: {e}")
    traceback.print_exc()
    exit(1)

# -----------------------------------------------------------------------------
# Main conversation loop (Unchanged)
# -----------------------------------------------------------------------------
def main():
    print("\n=== Welcome to the Multi-Level Banking Assistant (v9 - LLM Routing + Clarification) ===")
    # ... (main loop remains the same as v8) ...
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
        # top_matches_from_l2 is no longer needed in the state
    }

    while True:
        # Print results/errors from *previous* turn before asking for input
        final_task_result_prev_turn = current_state.get("task_result")
        if final_task_result_prev_turn:
            print(f"\nAssistant: {final_task_result_prev_turn}")

        final_error_message_prev_turn = current_state.get("error_message")
        if final_error_message_prev_turn:
             # ... (error message formatting - unchanged, handles LLM errors now too) ...
             if "Ambiguous request" in final_error_message_prev_turn:
                 print(f"\nAssistant: I'm still not sure what you meant. {final_error_message_prev_turn}. Could you please ask for 'help'?")
             elif "LLM parsing error" in final_error_message_prev_turn or "LLM Error" in final_error_message_prev_turn:
                  print(f"\nAssistant: Sorry, I had trouble understanding the options. Please try rephrasing or ask for 'help'. ({final_error_message_prev_turn})")
             elif final_error_message_prev_turn == "Authentication Required":
                  print(f"\nAssistant: Please log in first to complete your request.")
             elif final_error_message_prev_turn in ["Authentication Failed", "Account Not Found", "Email Exists"]:
                  print(f"\nAssistant: There was an issue with authentication: {final_error_message_prev_turn}. Please try again.")
             elif "Invalid" in final_error_message_prev_turn or "Missing" in final_error_message_prev_turn:
                  print(f"\nAssistant: There was an input error: {final_error_message_prev_turn}. Please try again.")
             elif "Internal error" in final_error_message_prev_turn or "Routing Decision" in final_error_message_prev_turn:
                   print(f"\nAssistant: Sorry, an internal routing error occurred. Please try again or ask for 'help'. ({final_error_message_prev_turn})")
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
            # top_matches_from_l2 REMOVED
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