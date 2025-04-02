import sys
from transformers import pipeline
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage

# -----------------------------------------------------------------------------
# Initialize a Huggingface text-generation model.
# (For demonstration, GPT-2 is used. For more dialogue-friendly models, consider alternatives.)
model = pipeline("text-generation", model="gpt2")

# -----------------------------------------------------------------------------
# Local database (in-memory) to manage user accounts.
# Structure: { email: { "name": ..., "password": ... } }
local_db = {}

# -----------------------------------------------------------------------------
# Tools: Functions that perform the specific tasks.

def login_tool():
    """Login tool: verifies user credentials."""
    email = input("Enter your email: ")
    password = input("Enter your password: ")
    user = local_db.get(email)
    if user and user["password"] == password:
        print(f"Login successful! Welcome back, {user['name']}.\n")
    else:
        print("Invalid email or password. Please try again.\n")

def signup_tool():
    """Sign up tool: registers a new user account."""
    name = input("Enter your name: ")
    email = input("Enter your email: ")
    if email in local_db:
        print("This email is already registered. Try logging in.\n")
        return
    password = input("Enter your password: ")
    # Add the user to the local database
    local_db[email] = {"name": name, "password": password}
    print("Sign up successful! You can now log in.\n")

def password_reset_tool():
    """Password reset tool: resets a user’s password."""
    email = input("Enter your email: ")
    if email not in local_db:
        print("Email not found. Please sign up first.\n")
        return
    new_password = input("Enter your new password: ")
    local_db[email]["password"] = new_password
    print("Password reset successful! You can now log in with your new password.\n")

# -----------------------------------------------------------------------------
# Agent descriptions for context. These help the supervisor decide which agent to call.
AGENT_DESCRIPTIONS = {
    "login": "Handles account login by verifying user email and password.",
    "signup": "Handles new account registration by collecting user name, email, and password.",
    "password_reset": "Handles password reset by verifying email and updating the password."
}

# -----------------------------------------------------------------------------
# Supervisor node: routes requests to the appropriate agent.
def supervisor_node(state: dict) -> Command:
    """
    Supervisor node:
    - Displays the context (agent descriptions) for debugging/decision-making.
    - Accepts runtime user input.
    - Routes to the appropriate agent based on simple keyword matching.
    """
    user_request = input("What would you like to do? (Options: login, signup, password reset) \n")
    
    # Display agent context
    context = (
        "Agent options:\n"
        "1. login - " + AGENT_DESCRIPTIONS["login"] + "\n" +
        "2. signup - " + AGENT_DESCRIPTIONS["signup"] + "\n" +
        "3. password_reset - " + AGENT_DESCRIPTIONS["password_reset"] + "\n"
    )
    print("\n[Supervisor Context]\n" + context)
    
    request_lower = user_request.lower()
    if "login" in request_lower:
        print("[Supervisor] Routing to Login Agent...\n")
        return Command(goto="login")
    elif "sign" in request_lower:
        print("[Supervisor] Routing to Signup Agent...\n")
        return Command(goto="signup")
    elif "reset" in request_lower or "password" in request_lower:
        print("[Supervisor] Routing to Password Reset Agent...\n")
        return Command(goto="password_reset")
    else:
        print("Unrecognized command. Please enter 'login', 'signup', or 'password reset'.\n")
        # Stay in the supervisor until a valid command is received
        return Command(goto="supervisor")

# -----------------------------------------------------------------------------
# Agent nodes that execute their specific tool then return to supervisor.
def login_node(state: dict) -> Command:
    login_tool()
    return Command(goto="END")

def signup_node(state: dict) -> Command:
    signup_tool()
    return Command(goto="END")

def password_reset_node(state: dict) -> Command:
    password_reset_tool()
    return Command(goto="END")

# -----------------------------------------------------------------------------
# Build the state graph with LangGraph.
builder = StateGraph(dict)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("login", login_node)
builder.add_node("signup", signup_node)
builder.add_node("password_reset", password_reset_node)
graph = builder.compile()

# -----------------------------------------------------------------------------
# Main loop: continuously run the orchestrator.
def main():
    print("=== Welcome to the Multi-Agent Account Manager (LangGraph Version) ===")
    while True:
        # Execute the graph using the stream method.
        for _ in graph.stream({"messages": []}):
            pass  # We iterate through the stream to execute the graph.
        cont = input("Do you want to perform another action? (yes/no): ")
        if cont.strip().lower() != "yes":
            print("Exiting the system. Goodbye!")
            break

if __name__ == "__main__":
    main()