# supervisors.py
import json
import traceback
from typing import List, Dict, Any

# Import necessary components
from state import AppState # Requires BaseMessage, UserInfo internally
from constants import (
    L2_AUTH, L2_ACCOUNT, L2_SUPPORT, L3_HANDOFF,
    CLARIFICATION_NODE, DISPLAY_DETAILS_ACTION, END
)
# --- Import Updated Prompts and Context ---
from prompts import (
    L1_SYSTEM_PROMPT_TEMPLATE, L2_SYSTEM_PROMPT_TEMPLATE, # Correct prompt templates
    L1_PROMPT_CONTEXT, L2_AUTH_CONTEXT, L2_ACCOUNT_CONTEXT, L2_SUPPORT_CONTEXT # Correct context strings
)
from llm_config import llm # Import the configured LLM instance
# --- Import descriptions components needed for validation/clarification ---
from descriptions import get_node_description, l2_auth_tool_definitions, l2_account_tool_definitions, l2_support_tool_definitions, l1_route_definitions

# Import message types used in prompts/state
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage


# -----------------------------------------------------------------------------
# --- Level 1 Supervisor (LLM Based - Async) ---
# -----------------------------------------------------------------------------
async def l1_main_supervisor_llm(state: AppState) -> Dict[str, Any]:
    """Routes user requests to L2 supervisors using LLM."""
    print("\n--- L1 Main Supervisor (LLM Based) ---")
    is_logged_in = bool(state.get('user_info'))
    auth_status = "Logged In" if is_logged_in else "Not Logged In"
    print(f"Current Auth Status: {auth_status}")

    messages = state.get('messages', [])
    if not messages or not isinstance(messages[-1], HumanMessage):
        print("L1 LLM Warning: No user message found. Routing to Support.")
        return {"next_action": L2_SUPPORT, "current_task": "Error: No user message", "error_message": "Missing User Input"}

    last_user_message = messages[-1].content
    print(f"User Message: '{last_user_message}'")

    # Prepare prompt using the correct context string from prompts.py
    system_prompt = L1_SYSTEM_PROMPT_TEMPLATE.format(l2_supervisor_descriptions=L1_PROMPT_CONTEXT)
    prompt_messages = [SystemMessage(content=system_prompt)]
    prompt_messages.append(SystemMessage(content=f"Current Authentication Status: {auth_status}"))
    prompt_messages.extend(messages[-5:])

    next_action = L2_SUPPORT # Default fallback
    error_message = None

    try:
        response = await llm.ainvoke(prompt_messages)
        llm_decision = response.content.strip()
        # Remove potential markdown code fences around the decision
        if llm_decision.startswith("```") and llm_decision.endswith("```"):
             llm_decision = llm_decision[3:-3].strip()
        # Remove potential JSON quotes if the LLM accidentally added them
        llm_decision = llm_decision.strip('"')

        print(f"L1 LLM Decision: {llm_decision}")

        # Validate against names from descriptions.py
        valid_l2_routes = [r["name"] for r in l1_route_definitions]
        if llm_decision in valid_l2_routes:
            if llm_decision == L2_ACCOUNT and not is_logged_in:
                print("L1 LLM Rule Applied: Account task requires login. Routing to Auth.")
                next_action = L2_AUTH
            else:
                next_action = llm_decision
        else:
            print(f"L1 LLM Warning: Invalid route '{llm_decision}' received. Defaulting to Support.")
            error_message = "L1 routing failed (invalid LLM response)"
            next_action = L2_SUPPORT

    except Exception as e:
        print(f"L1 LLM Error: Exception during LLM call: {e}")
        traceback.print_exc()
        error_message = f"L1 LLM Error: {e}"
        next_action = L2_SUPPORT

    print(f"L1 Final Routing Decision: {next_action}")
    return {
        "next_action": next_action,
        "current_task": last_user_message,
        "error_message": error_message,
        "suggested_choices": None,
    }

# -----------------------------------------------------------------------------
# --- Level 2 Supervisors (LLM Based - Async) ---
# -----------------------------------------------------------------------------

async def l2_llm_router_logic(
    state: AppState,
    supervisor_name: str,
    tool_definitions: List[Dict], # Pass the definitions list for validation
    tool_context: str, # Pass the pre-formatted context string for the prompt
    fallback_node: str = L3_HANDOFF
) -> Dict[str, Any]:
    """Generic L2 logic using LLM to find relevant L3 tools."""
    print(f"\n--- {supervisor_name} (LLM Based) ---")
    task = state.get("current_task", "")
    print(f"Received Task: '{task}'")

    # Login Check (Account Supervisor)
    if supervisor_name == L2_ACCOUNT and not state.get("user_info"):
        print("L2 Account: Authentication required. Routing to L2 Auth Supervisor.")
        return {"next_action": L2_AUTH, "current_task": f"login (required for: {task or 'your request'})", "error_message": "Authentication Required"}

    # Pre-check errors (Support Supervisor)
    if supervisor_name == L2_SUPPORT:
        error_msg_in = state.get("error_message")
        specific_errors = ["FAQ Not Found", "Unknown", "Unclear", "Failed", "Invalid", "Error", "Could not understand request"]
        if error_msg_in and any(err_pattern in error_msg_in for err_pattern in specific_errors):
            print(f"L2 Support: Handling specific error '{error_msg_in}', forcing handoff.")
            return {"next_action": L3_HANDOFF, "error_message": None} # Clear error before handoff

    # Call LLM
    next_action = fallback_node
    error_message = None
    suggested_choices = None

    try:
        # Use the correct prompt template and context string
        system_prompt = L2_SYSTEM_PROMPT_TEMPLATE.format(
            supervisor_name=supervisor_name, l3_tool_descriptions=tool_context, user_request=task)
        prompt_messages = [SystemMessage(content=system_prompt)]

        response = await llm.ainvoke(prompt_messages)
        llm_output_str = response.content.strip()
        print(f"{supervisor_name} LLM Decision Raw: {llm_output_str}")

        relevant_tools = []
        try:
            # Handle potential markdown code fences ```json ... ```
            if llm_output_str.startswith("```json"):
                llm_output_str = llm_output_str[7:]
                if llm_output_str.endswith("```"):
                    llm_output_str = llm_output_str[:-3]
                llm_output_str = llm_output_str.strip()

            parsed_output = json.loads(llm_output_str)

            if isinstance(parsed_output, list):
                # Get valid tool names from the definitions passed to this function
                valid_tools = [t["name"] for t in tool_definitions]
                for item in parsed_output:
                    # Validate structure and ensure name is valid for THIS supervisor
                    if (isinstance(item, dict) and "name" in item and "score" in item and
                        item["name"] in valid_tools and # Check against valid_tools
                        isinstance(item["score"], (int, float)) and 0.0 <= item["score"] <= 1.0):
                        # Score filtering is done by the prompt rule (e.g., >= 0.2)
                        # We trust the LLM followed the prompt here, but could add a filter if needed:
                        # if item["score"] >= 0.2: # Match the prompt threshold
                        relevant_tools.append(item)
                    else:
                        print(f"Warning: Invalid item format or name in LLM output: {item}")
                relevant_tools.sort(key=lambda x: x["score"], reverse=True)
            else:
                 raise ValueError("LLM output is not a JSON list.")

        except (json.JSONDecodeError, ValueError, TypeError) as json_e:
            print(f"{supervisor_name} LLM Error: Failed to parse JSON output: {json_e} | Output: '{llm_output_str}'")
            error_message = f"{supervisor_name} LLM parsing error."
            # next_action remains fallback_node

        print(f"{supervisor_name} Relevant Tools Found by LLM (>=0.2 score): {relevant_tools}")

        # --- Apply Routing Logic based on number of relevant tools ---
        num_relevant = len(relevant_tools)
        if num_relevant == 0:
            print(f"{supervisor_name}: No relevant tools found by LLM (or score < 0.2). Routing to fallback {fallback_node}.")
            error_message = error_message or f"No suitable tool found in {supervisor_name}."
            next_action = fallback_node
        elif num_relevant == 1:
            tool_name = relevant_tools[0]["name"]
            print(f"{supervisor_name}: Exactly one relevant tool found ({tool_name}). Routing directly.")
            next_action = tool_name
            error_message = None
        elif num_relevant == 2:
            print(f"{supervisor_name}: Two relevant tools found. Routing to clarification.")
            next_action = CLARIFICATION_NODE
            # Use get_node_description (imported) for user-friendly text
            suggested_choices = [(t["name"], get_node_description(t["name"])) for t in relevant_tools]
            error_message = None
        else: # More than 2 relevant tools
            print(f"{supervisor_name}: More than two relevant tools found. Routing to clarification (Top 3).")
            next_action = CLARIFICATION_NODE
            suggested_choices = [(t["name"], get_node_description(t["name"])) for t in relevant_tools[:3]] # Show top 3
            error_message = None

    except Exception as llm_e:
        print(f"{supervisor_name} LLM Error: Exception during LLM call: {llm_e}")
        traceback.print_exc()
        error_message = f"{supervisor_name} LLM Error: {llm_e}"
        next_action = fallback_node

    # Handle Account Supervisor's special display action case
    if supervisor_name == L2_ACCOUNT and next_action == DISPLAY_DETAILS_ACTION:
         print("L2 Account: Handling display details action directly.")
         user_info = state.get("user_info")
         if not user_info:
              return {"next_action": L2_AUTH, "error_message": "Authentication Required (Error in L2 Account)"}
         details = (f"Login/Display Name: {user_info.get('name', 'N/A')}\n"
                   f"Account Holder Name: {user_info.get('account_holder_name', 'N/A')}\n"
                   f"Account Number: {user_info.get('account_number', 'N/A')}\n"
                   f"Account Type: {user_info.get('account_type', 'N/A')}\n"
                   f"Account ID: {user_info.get('account_id', 'N/A')}")
         print(details + "\n")
         return {"task_result": details, "next_action": END, "error_message": None, "suggested_choices": None}

    print(f"{supervisor_name} Final Routing Decision: {next_action}")
    return {"next_action": next_action, "suggested_choices": suggested_choices, "error_message": error_message}

# Define L2 Supervisor nodes using the async helper
# Pass the correct context strings and definitions lists
async def l2_auth_supervisor(state: AppState) -> dict:
    return await l2_llm_router_logic(state, L2_AUTH, l2_auth_tool_definitions, L2_AUTH_CONTEXT, fallback_node=L2_SUPPORT)

async def l2_account_supervisor(state: AppState) -> dict:
    return await l2_llm_router_logic(state, L2_ACCOUNT, l2_account_tool_definitions, L2_ACCOUNT_CONTEXT, fallback_node=L2_SUPPORT)

async def l2_support_supervisor(state: AppState) -> dict:
    return await l2_llm_router_logic(state, L2_SUPPORT, l2_support_tool_definitions, L2_SUPPORT_CONTEXT, fallback_node=L3_HANDOFF)