import sys
from transformers import pipeline

# -----------------------------------------------------------------------------
# Initialize a Huggingface text-generation model.
# (Note: GPT-2 is used here for demonstration purposes. In practice, you may
# choose a model that better supports dialogue.)
model = pipeline("text-generation", model="gpt2")

# -----------------------------------------------------------------------------
# Local database (in-memory) to manage user accounts.
# Structure: { email: { "name": ..., "password": ... } }
local_db = {}

# -----------------------------------------------------------------------------
# Tools: Agent functions that perform the specific tasks.

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
# Each agent is defined with a description to help the supervisor decide.
AGENT_DESCRIPTIONS = {
    "login": "Handles account login by verifying user email and password.",
    "signup": "Handles new account registration by collecting user name, email, and password.",
    "password_reset": "Handles password reset by verifying email and updating the password."
}

# -----------------------------------------------------------------------------
# Supervisor agent that routes the user request to the appropriate agent.
def supervisor_agent():
    """
    Supervisor node:
    - Provides the agent descriptions (context) to decide which agent to route the request.
    - Accepts user input at runtime.
    - Delegates the task to the corresponding agent tool.
    """
    # Get the user request.
    user_request = input(
        "What would you like to do? (Options: login, signup, password reset) \n"
    )

    # (Optionally, one could use the Huggingface model to “interpret” the request by
    # generating text based on a prompt that includes the agent descriptions. Here, for
    # simplicity, we use simple keyword matching.)

    # Build a context string for debugging/decision-making:
    context = (
        "Agent options:\n"
        "1. login - " + AGENT_DESCRIPTIONS["login"] + "\n" +
        "2. signup - " + AGENT_DESCRIPTIONS["signup"] + "\n" +
        "3. password_reset - " + AGENT_DESCRIPTIONS["password_reset"] + "\n"
    )
    print("\n[Supervisor Context]\n" + context)
    
    # Simple decision logic based on keywords.
    request_lower = user_request.lower()
    if "login" in request_lower:
        print("[Supervisor] Routing to Login Agent...\n")
        login_tool()
    elif "sign" in request_lower:
        print("[Supervisor] Routing to Signup Agent...\n")
        signup_tool()
    elif "reset" in request_lower or "password" in request_lower:
        print("[Supervisor] Routing to Password Reset Agent...\n")
        password_reset_tool()
    else:
        print("Unrecognized command. Please enter 'login', 'signup', or 'password reset'.\n")

# -----------------------------------------------------------------------------
# Main loop: the orchestrator continuously accepts runtime input.
def main():
    print("=== Welcome to the Multi-Agent Account Manager ===")
    while True:
        supervisor_agent()
        cont = input("Do you want to perform another action? (yes/no): ")
        if cont.strip().lower() != "yes":
            print("Exiting the system. Goodbye!")
            break

if __name__ == "__main__":
    main()
