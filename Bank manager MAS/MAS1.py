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
# Local in-memory database
local_db = {
    "ayush@gmail.com": {"name": "ayush", "password": "123", "balance": 1500.75, "history": ["+ $1000 (Initial Deposit)", "- $50 (Groceries)", "+ $600.75 (Salary)"]}
}

# Simple FAQ database
faq_db = {
    "hours": "Our bank branches are open Mon-Fri 9 AM to 5 PM. Online banking is available 24/7.",
    "contact": "You can call us at 1-800-BANKING or visit our website's contact page.",
    "locations": "We have branches in Pune and Gurugram. Use our online locator for specific addresses."
}

# -----------------------------------------------------------------------------
# State Definition
class UserInfo(TypedDict):
    email: str
    name: str

class AppState(TypedDict):
    messages: Sequence[BaseMessage]
    user_info: Optional[UserInfo]
    current_task: Optional[str]
    task_result: Optional[str]
    # Routing control - decides which L2/L3 node to call next
    next_action: Optional[str]
    # Store error messages for better handling
    error_message: Optional[str] # SIMPLIFY to Optional[str]

# -----------------------------------------------------------------------------
# Level 3: Specialist Tools / Nodes
# These nodes execute specific tasks and update the state.

def login_tool_node(state: AppState) -> dict:
    """Node to handle the login process."""
    print("--- Executing Login Tool ---")
    email = input("Enter your email: ")
    password = input("Enter your password: ")
    user_data = local_db.get(email)
    if user_data and user_data["password"] == password:
        user_info = {"email": email, "name": user_data["name"]}
        result = f"Login successful! Welcome back, {user_info['name']}."
        print(result + "\n")
        # Update state with user info and clear any previous errors
        return {"user_info": user_info, "task_result": result, "error_message": None}
    else:
        result = "Invalid email or password. Please try again."
        print(result + "\n")
        # Return error state
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
    # Add basic balance and history for new users
    local_db[email] = {"name": name, "password": password, "balance": 0, "history": []}
    result = "Sign up successful! You can now log in."
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
    balance = local_db[email].get("balance", "N/A")
    result = f"Your current balance is: ${balance:.2f}"
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
    if history:
        history_str = "\n".join([f"- {item}" for item in history])
        result = f"Your recent transactions:\n{history_str}"
    else:
        result = "No transaction history found."
    print(result + "\n")
    return {"task_result": result, "error_message": None}

def faq_tool_node(state: AppState) -> dict:
    """Node to answer simple FAQs."""
    print("--- Executing FAQ Tool ---")
    # In a real system, the supervisor would identify keywords
    # For simplicity, we'll pick one based on the *last* user message content
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
        # If L1 routed here but no keywords match, let L2 Support handle it
        print("FAQ tool couldn't find a match, routing to support.")
        return {"next_action": "L2_SupportSupervisor", "error_message": "FAQ Not Found"}

    print(f"FAQ Result: {result}\n")
    return {"task_result": result, "error_message": None}


def human_handoff_node(state: AppState) -> dict:
    """Node to simulate handoff to a human agent."""
    print("--- Executing Human Handoff ---")
    result = "I understand this requires further assistance. Connecting you to a human agent now..."
    print(result + "\n")
    # In a real system, this would trigger an API call to a support platform
    return {"task_result": result, "next_action": END, "error_message": None} # End the graph flow

# -----------------------------------------------------------------------------
# Level 2: Departmental Supervisors (Simplified)
# These nodes decide which L3 tool to call based on L1's refined task
# For simplicity, using keyword matching here instead of another LLM call.

def l2_auth_supervisor(state: AppState) -> dict:
    """Supervisor for authentication tasks."""
    print("--- L2 Auth Supervisor ---")
    task = state.get("current_task", "").lower()
    if "login" in task or "sign in" in task:
        next_action = "L3_LoginToolNode"
    elif "signup" in task or "register" in task or "new account" in task or "create an account" in task:
        next_action = "L3_SignupToolNode"
    # Add password reset routing here if implemented
    # elif "password" in task or "reset" in task:
    #     next_action = "L3_PasswordResetToolNode"
    else:
        print(f"L2 Auth Warning: Unknown task '{task}'. Routing to Support.")
        # Default to support if task is unclear within this domain
        return {"next_action": "L2_SupportSupervisor", "error_message": "Unknown Auth Task"}
    print(f"Routing to: {next_action}")
    return {"next_action": next_action, "error_message": None}

def l2_account_supervisor(state: AppState) -> dict:
    """Supervisor for account management tasks."""
    print("--- L2 Account Supervisor ---")
    task = state.get("current_task", "").lower()

    # Check authentication before proceeding
    if not state.get("user_info"):
        print("Authentication required for account actions. Routing to Auth.")
        # Re-route to Auth supervisor if not logged in
        return {"next_action": "L2_AuthSupervisor", "current_task": "login (required for account access)", "error_message": "Authentication Required"}

    if "balance" in task:
        next_action = "L3_CheckBalanceToolNode"
    elif "history" in task or "transaction" in task or "statement" in task:
        next_action = "L3_GetHistoryToolNode"
    # Add other account actions (e.g., open account, card management) here
    else:
        print(f"L2 Account Warning: Unknown task '{task}'. Routing to Support.")
        return {"next_action": "L2_SupportSupervisor", "error_message": "Unknown Account Task"}
    print(f"Routing to: {next_action}")
    return {"next_action": next_action, "error_message": None}

def l2_support_supervisor(state: AppState) -> dict:
    """Supervisor for general support, fallback, and FAQs."""
    print("--- L2 Support Supervisor ---")
    task = state.get("current_task", "").lower()
    last_user_message = ""
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content.lower()
            break

    # Simple keyword matching for common support requests
    if any(kw in task or kw in last_user_message for kw in ["hour", "open", "contact", "phone", "call", "location", "address", "branch"]):
         next_action = "L3_FAQToolNode"
    elif any(kw in task or kw in last_user_message for kw in ["help", "support", "issue", "human", "agent", "problem"]):
         next_action = "L3_HumanHandoffNode"
    else:
         # If it reached support and doesn't match FAQ/Handoff, try generic handoff
         print("L2 Support: Task unclear or unhandled, suggesting handoff.")
         next_action = "L3_HumanHandoffNode"

    print(f"Routing to: {next_action}")
    return {"next_action": next_action, "error_message": None}


# -----------------------------------------------------------------------------
# Level 1: Main Orchestrator / Supervisor
# Uses LLM to classify intent and route to the appropriate L2 supervisor.

L1_SUPERVISOR_SYSTEM_PROMPT = """You are the main orchestrator for a multi-agent banking assistant.
Your goal is to understand the user's request and route it to the correct department supervisor.
The available departments are:
1.  **Auth**: Handles login, signup, password reset requests.
2.  **Account**: Handles requests about checking balance, transaction history, account details (requires login).
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
Example User Request: "What are your hours?" -> Respond: L2_SupportSupervisor
Example User Request: "Check my transactions" (User Not Logged In) -> Respond: L2_AuthSupervisor
Example User Request: "I need help with something unusual" -> Respond: L2_SupportSupervisor
"""

def l1_main_supervisor(state: AppState) -> dict:
    """Main supervisor node using LLM to route to L2 supervisors."""
    print("\n--- L1 Main Supervisor ---")
    messages = state['messages']
    user_info = state.get('user_info')
    auth_status = "Logged In" if user_info else "Not Logged In"
    print(f"Current Auth Status: {auth_status}")

    # Prepare messages for LLM
    prompt_messages = [SystemMessage(content=L1_SUPERVISOR_SYSTEM_PROMPT)]
    # Add context about auth status for the LLM
    prompt_messages.append(SystemMessage(content=f"Current Authentication Status: {auth_status}"))
    # Add recent conversation history (limit length if needed)
    prompt_messages.extend(messages[-5:]) # Send last 5 messages for context

    try:
        response = llm.invoke(prompt_messages)
        llm_decision = response.content.strip()
        print(f"L1 LLM Decision: {llm_decision}")

        # Validate LLM output
        valid_routes = ["L2_AuthSupervisor", "L2_AccountSupervisor", "L2_SupportSupervisor"]
        if llm_decision in valid_routes:
            next_action = llm_decision
            # Store the user's intent as the current task for L2
            last_user_message = ""
            for msg in reversed(messages):
                 if isinstance(msg, HumanMessage):
                     last_user_message = msg.content
                     break
            current_task = last_user_message # Use the raw message as task description
            print(f"Routing to: {next_action} with task: '{current_task}'")
            # Clear previous results/errors before routing
            return {"next_action": next_action, "current_task": current_task, "task_result": None, "error_message": None}
        else:
            print(f"L1 Warning: LLM produced invalid route '{llm_decision}'. Defaulting to Support.")
            return {"next_action": "L2_SupportSupervisor", "current_task": "Unknown intent", "error_message": "L1 Routing Failed"}

    except Exception as e:
        print(f"L1 Error: Exception during LLM call: {e}")
        # Fallback to support on LLM error
        return {"next_action": "L2_SupportSupervisor", "current_task": "System error", "error_message": f"L1 LLM Error: {e}"}


# -----------------------------------------------------------------------------
# Graph Definition

# Conditional routing functions
def route_l1_decision(state: AppState) -> str:
    """Routes from L1 based on 'next_action'."""
    next_node = state.get("next_action")
    print(f"[Router] L1 Decision: Route to {next_node}")
    if next_node in ["L2_AuthSupervisor", "L2_AccountSupervisor", "L2_SupportSupervisor"]:
        return next_node
    elif state.get("error_message"): # If L1 itself errored
         return "L2_SupportSupervisor" # Default to support on error
    else:
        print("[Router] L1 Warning: Invalid next_action from L1. Defaulting to Support.")
        return "L2_SupportSupervisor"

def route_l2_decision(state: AppState) -> str:
    """Routes from L2 supervisors to L3 tools or other L2/END."""
    next_node = state.get("next_action")
    print(f"[Router] L2 Decision: Route to {next_node}")

    # Check if L2 decided to re-route (e.g., auth required, task unknown)
    if next_node in ["L2_AuthSupervisor", "L2_SupportSupervisor"]:
        return next_node
    # Check if L2 decided to end (e.g., handoff)
    elif next_node == END:
         return END
    # Check for valid L3 tool nodes
    elif next_node in ["L3_LoginToolNode", "L3_SignupToolNode", "L3_CheckBalanceToolNode", "L3_GetHistoryToolNode", "L3_FAQToolNode", "L3_HumanHandoffNode"]:
        return next_node
    else:
        print(f"[Router] L2 Warning: Invalid next_action '{next_node}' from L2. Defaulting to Support.")
        # If the L2 node failed to produce a valid next step, go to support
        return "L2_SupportSupervisor"


def route_after_tool(state: AppState) -> str:
    """Determines the next step after an L3 tool executes."""
    print(f"[Router] After Tool Execution...")
    error_message = state.get("error_message")
    # Check if the tool itself forced an end state (like L3_HumanHandoffNode)
    # The tool node must explicitly return {"next_action": END, ...} for this
    next_action_forced_by_tool = state.get("next_action")

    if next_action_forced_by_tool == END:
        print("Routing to END (forced by tool).")
        return END

    if error_message:
        print(f"Error detected: {error_message}. Routing back to L1 Supervisor for re-evaluation.")
        # Let L1 handle the error state on the *next* turn if necessary,
        # or potentially add a dedicated error handling node.
        # For now, ending the turn after showing the error might be okay too.
        # Let's stick to routing to L1 on error for now.
        return "L1_Supervisor" # Or choose END if you want errors to also stop the turn
    else:
        # *** THIS IS THE KEY CHANGE ***
        # If the tool was successful (no error), end the current graph execution.
        # The main loop will then print the task_result and prompt for new input.
        print("Tool executed successfully. Ending current turn.")
        return END # Signal to stop the stream for this input

# Build the graph
builder = StateGraph(AppState)

# Add Nodes
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

# Define Edges

# Start goes to L1 Supervisor
builder.add_edge(START, "L1_Supervisor")

# Conditional routing from L1 Supervisor
builder.add_conditional_edges(
    "L1_Supervisor",
    route_l1_decision,
    {
        "L2_AuthSupervisor": "L2_AuthSupervisor",
        "L2_AccountSupervisor": "L2_AccountSupervisor",
        "L2_SupportSupervisor": "L2_SupportSupervisor",
        # Add END condition if L1 decides task is complete? (Future enhancement)
    }
)

# Conditional routing from L2 Supervisors
L2_SUPERVISORS = ["L2_AuthSupervisor", "L2_AccountSupervisor", "L2_SupportSupervisor"]
L3_TOOLS = ["L3_LoginToolNode", "L3_SignupToolNode", "L3_CheckBalanceToolNode", "L3_GetHistoryToolNode", "L3_FAQToolNode", "L3_HumanHandoffNode"]

l2_conditional_map = {node: node for node in L3_TOOLS}
l2_conditional_map["L2_AuthSupervisor"] = "L2_AuthSupervisor" # For re-routing
l2_conditional_map["L2_SupportSupervisor"] = "L2_SupportSupervisor" # For re-routing
l2_conditional_map[END] = END # Allow L2 to signal END (for handoff)

for supervisor_node in L2_SUPERVISORS:
    builder.add_conditional_edges(
        supervisor_node,
        route_l2_decision,
        l2_conditional_map
    )

# Routing after L3 tools execute
# Routing after L3 tools execute
for tool_node in L3_TOOLS:
     # Use conditional edge to check for errors or forced END
     builder.add_conditional_edges(
         tool_node,
         route_after_tool,
         {
             "L1_Supervisor": "L1_Supervisor", # Route back to L1 ONLY if route_after_tool returns "L1_Supervisor" (e.g., on error)
             END: END                         # Route to END if route_after_tool returns END (e.g., on success or forced END)
         }
     )

# Compile the graph
try:
    graph = builder.compile()
    print("\nGraph compiled successfully!")
except Exception as e:
    print(f"\nFatal Error: Graph compilation failed: {e}")
    exit(1)

# Visualize (Optional - requires graphviz)
try:
    # Save graph visualization
    graph.get_graph().draw_mermaid_png(output_file_path="banking_agent_graph.png")
    print("Graph visualization saved to banking_agent_graph.png")
except ImportError:
    print("Install pygraphviz and graphviz to visualize the graph: pip install pygraphviz")
except Exception as e:
    print(f"Warning: Could not generate graph visualization: {e}")


# -----------------------------------------------------------------------------
# Main conversation loop

def main():
    print("\n=== Welcome to the Multi-Level Banking Assistant ===")
    print("Type 'quit' or 'exit' to end the conversation.")

    # Initialize state for a new conversation
    current_state: AppState = { # Add type hint for clarity
        "messages": [],
        "user_info": None,
        "current_task": None,
        "task_result": None,
        "next_action": None,
        "error_message": None,
    }

    while True:
        user_input = input("\nYou: ")
        if 'quit' in user_input.lower() or 'exit' in user_input.lower():
            print("Banking Assistant: Goodbye!")
            break

        # *** FIX: Update the messages list in the state dictionary ***
        current_messages = current_state.get('messages', [])
        # Ensure messages is treated as a list before appending
        if not isinstance(current_messages, list):
             current_messages = list(current_messages) # Convert if needed (e.g., from tuple)

        current_state['messages'] = current_messages + [HumanMessage(content=user_input)]
        # Clear previous task result and error before the new run
        current_state['task_result'] = None
        current_state['error_message'] = None


        print("\nAssistant Processing...")
        try:
            # Stream events to see the flow
            final_state_update = None # Track the last update from the stream
            for event in graph.stream(current_state, {"recursion_limit": 25}):
                # The event key is the node name
                node_name = list(event.keys())[0]
                node_output = event[node_name]
                print(f"--- Event: Node '{node_name}' ---")
                # print(f"Raw Output: {node_output}") # Optional: for deep debugging

                # Accumulate state updates from the stream
                # Important: node_output contains *only the fields updated by that node*
                current_state.update(node_output)
                final_state_update = node_output # Remember the last change

            # *** FIX: Adjusted Output Logic ***
            # After the stream finishes for this turn, check the final state
            if current_state.get("task_result"):
                 # Print the direct result from the last successful tool
                 print(f"\nAssistant: {current_state['task_result']}")
            elif current_state.get("error_message"):
                 # Or print the error if one occurred
                 print(f"\nAssistant: There was an issue: {current_state['error_message']}. Please clarify or try again.")
            # Add a generic prompt if the graph ended without a specific result/error
            # Avoid printing generic prompt if handoff occurred (it prints its own message)
            elif current_state.get("next_action") != END:
                  # Check if the *very last* message is already an AI one (e.g., from a node directly adding)
                 last_message = current_state['messages'][-1] if current_state.get('messages') else None
                 if not isinstance(last_message, AIMessage):
                     print("\nAssistant: How else can I help you?")


            # Check if graph ended via a node setting next_action to END
            # This might be slightly redundant if the node itself prints a final message
            # if current_state.get("next_action") == END:
            #    print("Assistant: Session ended.")


        except Exception as e:
            print(f"\n--- Critical Error during graph execution ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("\nAssistant: I've encountered a system error and cannot continue. Please try again later.")
            # Optionally break the loop on critical errors
            break

if __name__ == "__main__":
    main()