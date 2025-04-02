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
def llm_decide(user_request: str, agent_context: str) -> str:
    """
    Uses the Huggingface model to decide which agent to call.
    Returns one of: "login", "signup", "password_reset", or "unrecognized".
    """
    prompt = (
        f"You are a routing assistant for a multi-agent account manager.\n"
        f"User request: {user_request}\n"
        f"Agent options:\n{agent_context}\n"
        "Based on the user request and agent context, decide which agent to use. "
        "Respond with only one word from: login, signup, password_reset."
    )
    # Use max_new_tokens and truncation=True to generate new tokens without truncation issues.
    response = model(prompt, max_new_tokens=50, truncation=True)[0]['generated_text']
    response_lower = response.lower()
    
    if "login" in response_lower:
        return "login"
    elif "signup" in response_lower:
        return "signup"
    elif "password_reset" in response_lower or "reset" in response_lower or "password" in response_lower:
        return "password_reset"
    else:
        return "unrecognized"

def supervisor_node(state: dict) -> Command:
    """
    Supervisor node that:
    - Accepts runtime user input.
    - Displays the agent context.
    - Uses the LLM to decide which agent to call.
    """
    user_request = input("What would you like to do? (Options: login, signup, password reset) \n")
    
    # Build agent context for the LLM.
    agent_context = (
        "1. login - " + AGENT_DESCRIPTIONS["login"] + "\n" +
        "2. signup - " + AGENT_DESCRIPTIONS["signup"] + "\n" +
        "3. password_reset - " + AGENT_DESCRIPTIONS["password_reset"] + "\n"
    )
    print("\n[Supervisor Context]\n" + agent_context)
    
    # Use LLM to decide on the agent.
    decision = llm_decide(user_request, agent_context)
    print(f"[Supervisor] LLM decision: {decision}\n")
    
    if decision == "login":
        return Command(goto="login")
    elif decision == "signup":
        return Command(goto="signup")
    elif decision == "password_reset":
        return Command(goto="password_reset")
    else:
        print("Unrecognized command by LLM. Please enter a valid option.\n")
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