import random
import string
import uuid
from typing import TypedDict, List, Optional, Sequence, Annotated
from LLMaas import LLMaaSModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Create an instance of LLMaaSModel and get the LLM.
try:
    llmaas_model_instance = LLMaaSModel()
    llm = llmaas_model_instance.get_model()
    print(f"Using LLMaaS model: {llm.model_name}")
except Exception as e:
    print(f"Fatal Error: Could not initialize LLMaaS model: {e}")
    exit(1)

# -----------------------------------------------------------------------------
# Helper functions for new account details
def generate_account_number():
    """Generates a simple random account number."""
    return ''.join(random.choices(string.digits, k=10))

def generate_account_id():
    """Generates a unique account ID."""
    return str(uuid.uuid4())

# -----------------------------------------------------------------------------
# Local in-memory database - ADDED new fields
local_db = {
    "ayush@gmail.com": {
        "name": "ayush",
        "password": "123",
        "balance": 1500.75,
        "history": ["+ $1000 (Initial Deposit)", "- $50 (Groceries)", "+ $600.75 (Salary)"],
        "account_number": generate_account_number(), # Added
        "account_id": generate_account_id(),      # Added
        "account_type": "Savings"                 # Added
    }
}

# Simple FAQ database (unchanged)
faq_db = {
    "hours": "Our bank branches are open Mon-Fri 9 AM to 5 PM. Online banking is available 24/7.",
    "contact": "You can call us at 1-800-BANKING or visit our website's contact page.",
    "locations": "We have branches in Pune and Gurugram. Use our online locator for specific addresses."
}

# -----------------------------------------------------------------------------
# State Definition - ADDED new fields to UserInfo
class UserInfo(TypedDict):
    email: str
    name: str
    account_number: str # Added
    account_id: str     # Added
    account_type: str   # Added

class AppState(TypedDict):
    messages: Sequence[BaseMessage]
    user_info: Optional[UserInfo]
    current_task: Optional[str]
    task_result: Optional[str]
    next_action: Optional[str]
    error_message: Optional[str]

# -----------------------------------------------------------------------------
# Level 3: Specialist Tools / Nodes

# --- Existing Tools (Modified Login/Signup) ---

def login_tool_node(state: AppState) -> dict:
    """Node to handle the login process and load user info into state."""
    print("--- Executing Login Tool ---")
    email = input("Enter your email: ")
    password = input("Enter your password: ")
    user_data = local_db.get(email)
    if user_data and user_data["password"] == password:
        # *** UPDATED: Populate full UserInfo ***
        user_info: UserInfo = {
            "email": email,
            "name": user_data["name"],
            "account_number": user_data.get("account_number", "N/A"), # Get new fields
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
    name = input("Enter your name: ")
    email = input("Enter your email: ")
    if email in local_db:
        result = "This email is already registered. Try logging in."
        print(result + "\n")
        return {"task_result": result, "error_message": "Email Exists"}

    password = input("Enter your password: ")
    # *** UPDATED: Add new account details for new users ***
    account_number = generate_account_number()
    account_id = generate_account_id()
    account_type = "Savings" # Default to Savings for new signups

    local_db[email] = {
        "name": name,
        "password": password,
        "balance": 0,
        "history": [],
        "account_number": account_number,
        "account_id": account_id,
        "account_type": account_type
    }
    result = f"Sign up successful! Your new {account_type} account ({account_number}) is ready. You can now log in."
    print(result + "\n")
    return {"task_result": result, "error_message": None}

def check_balance_tool_node(state: AppState) -> dict:
    """Node to check account balance."""
    print("--- Executing Check Balance Tool ---")
    if not state.get("user_info"):
        result = "Error: You must be logged in to check your balance."
        print(result + "\n")
        return {"task_result": result, "error_message": "Authentication Required"}

    email = state["user_info"]["email"]
    # Defensive check if balance exists
    balance = local_db[email].get("balance", None)
    if balance is None:
         result = "Error: Could not retrieve balance information."
         print(result + "\n")
         return {"task_result": result, "error_message": "Balance Data Missing"}

    account_number = state["user_info"].get("account_number", "N/A") # Get from state
    result = f"Your current balance for account {account_number} is: ${balance:.2f}"
    print(result + "\n")
    return {"task_result": result, "error_message": None}

def get_history_tool_node(state: AppState) -> dict:
    """Node to retrieve transaction history."""
    print("--- Executing Get History Tool ---")
    if not state.get("user_info"):
        result = "Error: You must be logged in to view transaction history."
        print(result + "\n")
        return {"task_result": result, "error_message": "Authentication Required"}

    email = state["user_info"]["email"]
    history = local_db[email].get("history", [])
    account_number = state["user_info"].get("account_number", "N/A") # Get from state

    if history:
        history_str = "\n".join([f"- {item}" for item in history])
        result = f"Your recent transactions for account {account_number}:\n{history_str}"
    else:
        result = f"No transaction history found for account {account_number}."
    print(result + "\n")
    return {"task_result": result, "error_message": None}

def faq_tool_node(state: AppState) -> dict:
    """Node to answer simple FAQs."""
    print("--- Executing FAQ Tool ---")
    last_user_message = ""
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content.lower()
            break

    if "hour" in last_user_message or "open" in last_user_message:
        result = faq_db.get("hours", "Sorry, I don't have information on opening hours.")
    elif "contact" in last_user_message or "phone" in last_user_message or "call" in last_user_message:
         result = faq_db.get("contact", "Sorry, I don't have contact information.")
    elif "location" in last_user_message or "address" in last_user_message or "branch" in last_user_message:
         result = faq_db.get("locations", "Sorry, I don't have location information.")
    else:
        print("FAQ tool couldn't find a match, routing to support.")
        # *** CHANGED: No next_action here, let route_after_tool handle error ***
        return {"task_result": "I couldn't find a direct answer in the FAQ.", "error_message": "FAQ Not Found"}

    print(f"FAQ Result: {result}\n")
    return {"task_result": result, "error_message": None}


def human_handoff_node(state: AppState) -> dict:
    """Node to simulate handoff to a human agent."""
    print("--- Executing Human Handoff ---")
    result = "I understand this requires further assistance. Connecting you to a human agent now..."
    print(result + "\n")
    return {"task_result": result, "next_action": END, "error_message": None}

# --- NEW Tools ---

def password_reset_tool_node(state: AppState) -> dict:
    """Node to handle password reset (simulated)."""
    print("--- Executing Password Reset Tool ---")
    # In a real system, this requires verification (email link, OTP, etc.)
    # For simulation, we just ask for the email and new password.
    email = input("Enter the email for the account to reset password: ")
    if email not in local_db:
        result = f"Error: No account found with the email '{email}'."
        print(result + "\n")
        return {"task_result": result, "error_message": "Account Not Found"}

    new_password = input(f"Enter the new password for {email}: ")
    # Add basic validation if desired (e.g., length)
    if not new_password or len(new_password) < 3: # Basic check
        result = "Error: New password is too short. Please try again."
        print(result + "\n")
        return {"task_result": result, "error_message": "Invalid New Password"}

    local_db[email]["password"] = new_password
    result = f"Password for {email} has been updated successfully."
    print(result + "\n")
    # Usually, you'd force the user to log in again after reset.
    # For simplicity here, we just update it. We don't log them in.
    return {"task_result": result, "error_message": None}


def name_update_tool_node(state: AppState) -> dict:
    """Node to update the user's name."""
    print("--- Executing Name Update Tool ---")
    if not state.get("user_info"):
        result = "Error: You must be logged in to update your name."
        print(result + "\n")
        return {"task_result": result, "error_message": "Authentication Required"}

    user_info = state["user_info"]
    email = user_info["email"]

    current_name = user_info["name"]
    new_name = input(f"Your current name is '{current_name}'. Enter the new name: ")

    if not new_name or new_name.strip() == "":
        result = "Error: New name cannot be empty."
        print(result + "\n")
        return {"task_result": result, "error_message": "Invalid New Name"}

    new_name = new_name.strip()

    if email in local_db:
        local_db[email]["name"] = new_name
        # *** IMPORTANT: Update the name in the current state as well ***
        updated_user_info = user_info.copy() # Create a copy to modify
        updated_user_info["name"] = new_name

        result = f"Your name has been updated to '{new_name}'."
        print(result + "\n")
        # Return the updated user_info so the state reflects the change immediately
        return {"task_result": result, "user_info": updated_user_info, "error_message": None}
    else:
        # This shouldn't happen if user_info is present, but good practice
        result = "Error: Could not find your account details to update name."
        print(result + "\n")
        # Log out the user if their state doesn't match DB? (More complex handling)
        return {"task_result": result, "error_message": "Account Data Mismatch", "user_info": None}


# -----------------------------------------------------------------------------
# Level 2: Departmental Supervisors (Updated)

def l2_auth_supervisor(state: AppState) -> dict:
    """Supervisor for authentication tasks."""
    print("--- L2 Auth Supervisor ---")
    task = state.get("current_task", "").lower()
    if "login" in task or "sign in" in task:
        next_action = "L3_LoginToolNode"
    elif "signup" in task or "register" in task or "new account" in task or "create an account" in task:
        next_action = "L3_SignupToolNode"
    # *** ADDED: Password reset routing ***
    elif "password" in task or "reset" in task or "forgot" in task:
        next_action = "L3_PasswordResetToolNode"
    else:
        print(f"L2 Auth Warning: Unknown task '{task}'. Routing to Support.")
        return {"next_action": "L2_SupportSupervisor", "error_message": "Unknown Auth Task"}
    print(f"Routing to: {next_action}")
    return {"next_action": next_action, "error_message": None}

def l2_account_supervisor(state: AppState) -> dict:
    """Supervisor for account management tasks."""
    print("--- L2 Account Supervisor ---")
    task = state.get("current_task", "").lower()

    if not state.get("user_info"):
        print("Authentication required for account actions. Routing to Auth.")
        # Modify current_task to guide login process
        original_task = state.get("current_task", "your request")
        return {
            "next_action": "L2_AuthSupervisor",
            "current_task": f"login (required for: {original_task})",
            "error_message": "Authentication Required"
        }

    if "balance" in task:
        next_action = "L3_CheckBalanceToolNode"
    elif "history" in task or "transaction" in task or "statement" in task:
        next_action = "L3_GetHistoryToolNode"
    # *** ADDED: Name update routing ***
    elif "update my name" in task or "change my name" in task or "correct my name" in task or "correct name" in task or "change name" in task:
        next_action = "L3_NameUpdateToolNode"
    # Add other account actions (e.g., view account details, transfer funds) here
    # Example: Add routing for viewing account details
    elif "account number" in task or "account id" in task or "account type" in task or "account details" in task:
        # We can add a specific tool or just display info here
        user_info = state["user_info"]
        details = (
            f"Account Holder: {user_info['name']}\n"
            f"Account Number: {user_info['account_number']}\n"
            f"Account Type: {user_info['account_type']}\n"
            f"Account ID: {user_info['account_id']}"
        )
        print("--- L2 Account Supervisor: Displaying Details ---")
        print(details + "\n")
        # Since this supervisor handles it directly, end the turn
        return {"task_result": details, "next_action": END, "error_message": None}
    else:
        print(f"L2 Account Warning: Unknown task '{task}'. Routing to Support.")
        return {"next_action": "L2_SupportSupervisor", "error_message": "Unknown Account Task"}

    print(f"Routing to: {next_action}")
    return {"next_action": next_action, "error_message": None}


# L2 Support Supervisor remains the same conceptually, but its fallback role becomes more critical
def l2_support_supervisor(state: AppState) -> dict:
    """Supervisor for general support, fallback, and FAQs."""
    print("--- L2 Support Supervisor ---")
    task = state.get("current_task", "").lower()
    last_user_message = ""
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content.lower()
            break

    # Check if previous step resulted in an error handled by support (like FAQ not found)
    if state.get("error_message") == "FAQ Not Found":
        print("Handling 'FAQ Not Found' error, suggesting handoff.")
        next_action = "L3_HumanHandoffNode"

    # Simple keyword matching for common support requests
    elif any(kw in task or kw in last_user_message for kw in ["hour", "open", "contact", "phone", "call", "location", "address", "branch"]):
         next_action = "L3_FAQToolNode"
    elif any(kw in task or kw in last_user_message for kw in ["help", "support", "issue", "human", "agent", "problem", "talk to someone"]):
         next_action = "L3_HumanHandoffNode"
    else:
         print("L2 Support: Task unclear or unhandled, suggesting handoff.")
         next_action = "L3_HumanHandoffNode"

    print(f"Routing to: {next_action}")
    # Clear the specific error if we handled it (e.g., FAQ Not Found)
    error_msg = state.get("error_message") if state.get("error_message") != "FAQ Not Found" else None
    return {"next_action": next_action, "error_message": error_msg}


# -----------------------------------------------------------------------------
# Level 1: Main Orchestrator / Supervisor - UPDATED PROMPT

L1_SUPERVISOR_SYSTEM_PROMPT = """You are the main orchestrator for a multi-agent banking assistant.
Your goal is to understand the user's request and route it to the correct department supervisor.
The available departments are:
1.  **Auth**: Handles login, signup, **password reset/forgot password** requests.
2.  **Account**: Handles requests about checking balance, transaction history, account details (like account number, type), **updating or changing personal details like name**. Requires login.
3.  **Support**: Handles general questions (FAQs like hours, contact), requests for human help, or issues not covered by other departments.

Based on the user's latest message and the conversation history, determine the primary intent.
Consider if the user is currently authenticated (logged in). Some departments require authentication.

Respond *only* with the name of the department supervisor to route to:
- L2_AuthSupervisor
- L2_AccountSupervisor
- L2_SupportSupervisor

If the user needs to be logged in for their request but isn't, route to L2_AuthSupervisor first.
If the request is very generic or asks for help, route to L2_SupportSupervisor.

Example User Request: "What's my balance?" (User Logged In) -> Respond: L2_AccountSupervisor
Example User Request: "I forgot my password" -> Respond: L2_AuthSupervisor
Example User Request: "Change/update my name" (User Logged In) -> Respond: L2_AccountSupervisor
Example User Request: "What are your hours?" -> Respond: L2_SupportSupervisor
Example User Request: "Check my transactions" (User Not Logged In) -> Respond: L2_AuthSupervisor
Example User Request: "I need help resetting my password" -> Respond: L2_AuthSupervisor
Example User Request: "Show my account number" (User Logged In) -> Respond: L2_AccountSupervisor
Example User Request: "I want to update my name" (User Not Logged In) -> Respond: L2_AuthSupervisor
Example User Request: "I need help with something unusual" -> Respond: L2_SupportSupervisor
"""

# l1_main_supervisor function remains the same logic, just uses the updated prompt
def l1_main_supervisor(state: AppState) -> dict:
    """Main supervisor node using LLM to route to L2 supervisors."""
    print("\n--- L1 Main Supervisor ---")
    messages = state['messages']
    user_info = state.get('user_info')
    auth_status = "Logged In" if user_info else "Not Logged In"
    print(f"Current Auth Status: {auth_status}")

    prompt_messages = [SystemMessage(content=L1_SUPERVISOR_SYSTEM_PROMPT)]
    prompt_messages.append(SystemMessage(content=f"Current Authentication Status: {auth_status}"))
    # Limit history to prevent exceeding token limits, include last 5 messages
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
            # Try to capture the user message even on failure
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
# Graph Definition - UPDATED with new nodes and routing

# Conditional routing functions (route_l1_decision, route_l2_decision, route_after_tool) remain the same logic

def route_l1_decision(state: AppState) -> str:
    """Routes from L1 based on 'next_action'."""
    next_node = state.get("next_action")
    print(f"[Router] L1 Decision: Route to {next_node}")
    if next_node in ["L2_AuthSupervisor", "L2_AccountSupervisor", "L2_SupportSupervisor"]:
        return next_node
    elif state.get("error_message"): # If L1 itself errored
         print("[Router] L1 Error detected. Defaulting to Support.")
         return "L2_SupportSupervisor" # Default to support on error
    else:
        print(f"[Router] L1 Warning: Invalid next_action '{next_node}' from L1. Defaulting to Support.")
        # Ensure error state is updated if routing fails unexpectedly
        state["error_message"] = state.get("error_message", "Invalid L1 Routing")
        return "L2_SupportSupervisor"

def route_l2_decision(state: AppState) -> str:
    """Routes from L2 supervisors to L3 tools or other L2/END."""
    next_node = state.get("next_action")
    print(f"[Router] L2 Decision: Route to {next_node}")

    # Define ALL valid L3 tool nodes
    valid_l3_nodes = [
        "L3_LoginToolNode", "L3_SignupToolNode", "L3_CheckBalanceToolNode",
        "L3_GetHistoryToolNode", "L3_FAQToolNode", "L3_HumanHandoffNode",
        "L3_PasswordResetToolNode", "L3_NameUpdateToolNode" # Added new tools
    ]
    # Define valid L2 rerouting targets
    valid_l2_reroutes = ["L2_AuthSupervisor", "L2_SupportSupervisor"]

    if next_node in valid_l3_nodes:
        return next_node
    elif next_node in valid_l2_reroutes:
        return next_node
    elif next_node == END:
         return END
    else:
        print(f"[Router] L2 Warning: Invalid next_action '{next_node}' from L2. Defaulting to Support.")
        state["error_message"] = state.get("error_message", f"Invalid L2 Routing: {next_node}")
        return "L2_SupportSupervisor"

def route_after_tool(state: AppState) -> str:
    """Determines the next step after an L3 tool executes."""
    print(f"[Router] After Tool Execution...")
    error_message = state.get("error_message")
    next_action_forced_by_tool = state.get("next_action")

    if next_action_forced_by_tool == END:
        print("Routing to END (forced by tool).")
        return END

    if error_message:
        print(f"Error detected: {error_message}.")
        # Decide where errors go. Going back to L1 might cause loops if the intent is misunderstood.
        # Let's try ending the turn on error for now, user can rephrase.
        # Alternative: Route to L2_SupportSupervisor for specific error handling
        if error_message in ["Authentication Required", "Authentication Failed"]:
             print("Authentication error. Routing to L2 Auth Supervisor.")
             # Clear the error message as L2 Auth will handle the flow
             state["error_message"] = None
             # Trigger login explicitly
             state["current_task"] = "login (required due to previous error)"
             return "L2_AuthSupervisor"
        elif error_message == "FAQ Not Found":
             print("FAQ not found. Routing to L2 Support Supervisor.")
             # Keep the error message for L2 Support to see
             return "L2_SupportSupervisor"
        else:
             print("Ending turn due to unhandled tool error.")
             return END # End the turn, main loop will display error
    else:
        # Successful tool execution ends the current turn.
        print("Tool executed successfully. Ending current turn.")
        return END


# Build the graph
builder = StateGraph(AppState)

# Add Nodes (including NEW ones)
builder.add_node("L1_Supervisor", l1_main_supervisor)
builder.add_node("L2_AuthSupervisor", l2_auth_supervisor)
builder.add_node("L2_AccountSupervisor", l2_account_supervisor)
builder.add_node("L2_SupportSupervisor", l2_support_supervisor)

builder.add_node("L3_LoginToolNode", login_tool_node)
builder.add_node("L3_SignupToolNode", signup_tool_node)
builder.add_node("L3_CheckBalanceToolNode", check_balance_tool_node)
builder.add_node("L3_GetHistoryToolNode", get_history_tool_node)
builder.add_node("L3_FAQToolNode", faq_tool_node)
builder.add_node("L3_HumanHandoffNode", human_handoff_node)
builder.add_node("L3_PasswordResetToolNode", password_reset_tool_node) # New
builder.add_node("L3_NameUpdateToolNode", name_update_tool_node)     # New

# Define Edges

builder.add_edge(START, "L1_Supervisor")

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
    "L3_LoginToolNode", "L3_SignupToolNode", "L3_CheckBalanceToolNode",
    "L3_GetHistoryToolNode", "L3_FAQToolNode", "L3_HumanHandoffNode",
    "L3_PasswordResetToolNode", "L3_NameUpdateToolNode" # Added new tools
]
L2_SUPERVISORS = ["L2_AuthSupervisor", "L2_AccountSupervisor", "L2_SupportSupervisor"]

# Update conditional map for L2 routing
l2_conditional_map = {node: node for node in L3_TOOLS}
l2_conditional_map["L2_AuthSupervisor"] = "L2_AuthSupervisor"
l2_conditional_map["L2_SupportSupervisor"] = "L2_SupportSupervisor"
l2_conditional_map[END] = END

for supervisor_node in L2_SUPERVISORS:
    builder.add_conditional_edges(
        supervisor_node,
        route_l2_decision,
        l2_conditional_map
    )

# Routing after L3 tools execute (using the updated L3_TOOLS list)
# The routing logic itself (`route_after_tool`) determines the destination (END, L1, L2)
after_tool_map = {
    "L1_Supervisor": "L1_Supervisor",    # Not typically used now, but kept for flexibility
    "L2_AuthSupervisor": "L2_AuthSupervisor", # Added for specific error re-routing
    "L2_SupportSupervisor": "L2_SupportSupervisor", # Added for specific error re-routing
    END: END                            # Default success/error path
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
    graph.get_graph().draw_mermaid_png(output_file_path="banking_agent_graph_v2.png")
    print("Graph visualization saved to banking_agent_graph_v2.png")
except ImportError:
    print("Install pygraphviz and graphviz to visualize the graph: pip install pygraphviz")
except Exception as e:
    print(f"Warning: Could not generate graph visualization: {e}")


# -----------------------------------------------------------------------------
# Main conversation loop (Unchanged from previous good version)

def main():
    print("\n=== Welcome to the Multi-Level Banking Assistant (v2) ===")
    print("You can ask about balance, history, FAQs, login, signup, password reset, name updates.")
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
        # Display current login status for context
        auth_display = f"(Logged in as: {current_state['user_info']['name']})" if current_state.get("user_info") else "(Not Logged In)"
        user_input = input(f"\nYou {auth_display}: ")

        if 'quit' in user_input.lower() or 'exit' in user_input.lower():
            print("Banking Assistant: Goodbye!")
            break

        # Prepare state for the new turn
        current_messages = current_state.get('messages', [])
        if not isinstance(current_messages, list):
             current_messages = list(current_messages) # Ensure it's a list

        # *** FIX: Update the messages list directly ***
        current_messages.append(HumanMessage(content=user_input))
        current_state['messages'] = current_messages # Assign the updated list back to the state

        # Clear previous turn's results/errors before invoking graph
        current_state['task_result'] = None
        current_state['error_message'] = None
        current_state['next_action'] = None # Important: clear next_action before run

        print("\nAssistant Processing...")
        try:
            # Stream updates and update the state progressively
            for event in graph.stream(current_state, {"recursion_limit": 25}):
                node_name = list(event.keys())[0]
                node_output = event[node_name]
                print(f"--- Event: Node '{node_name}' Output: {node_output} ---") # More verbose logging
                # Update the central state dictionary with the partial update from the node
                current_state.update(node_output)

            # After the stream finishes for this input, display the outcome
            final_task_result = current_state.get("task_result")
            final_error_message = current_state.get("error_message")

            if final_task_result:
                 print(f"\nAssistant: {final_task_result}")
                 # Optionally add the final task result as an AIMessage
                 # current_state = add_messages(current_state, [AIMessage(content=final_task_result)])
            elif final_error_message:
                 print(f"\nAssistant: There was an issue: {final_error_message}. Please clarify or try again.")
                 # Optionally add error as an AIMessage
                 # current_state = add_messages(current_state, [AIMessage(content=f"Error: {final_error_message}")])
            else:
                 # Check if the graph ended normally without a specific result (should be less common now)
                 # Avoid printing generic prompt if handoff occurred (it prints its own message and sets next_action=END)
                 if current_state.get("next_action") != END:
                    last_message = current_state['messages'][-1] if current_state.get('messages') else None
                    # Only add generic prompt if the last message wasn't already from the AI
                    if not isinstance(last_message, AIMessage):
                         print("\nAssistant: How else can I help you today?")
                         # current_state = add_messages(current_state, [AIMessage(content="How else can I help you today?")])


        except Exception as e:
            print(f"\n--- Critical Error during graph execution ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("\nAssistant: I've encountered a system error and cannot continue this request. Please try again later.")
            # Reset parts of state after critical error? Or break?
            current_state['error_message'] = "Critical System Error" # Log the error state
            # break # Optional: end conversation on critical error

if __name__ == "__main__":
    main()