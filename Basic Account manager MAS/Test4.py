import httpx
import langchain_openai
# from pathlib import Path
# import json
from typing import Literal
# from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

# -----------------------------------------------------------------------------
# Custom LLMaaS model definition.
class LLMaaSModel:
    """
    This is a class for LLMaaS models.
    """
    def __init__(self):
        # Hardcoded values from the provided JSON structure.
        self.client_id = "idp-99629305-eb32-4450-af6d-0762ac02ca2b-llmaas-app"
        self.client_secret = "cidps_YUy05kngKz8GhSMFp4ghXHsHbiZt2h9BQsb6K4pP8zlTdF1Q9D8rTeHgbCFKmQJ5USqO541ogcf7cdHxDFTh5EQr2DCcz9"
        self.grant_type = "client_credentials"
        self.url = "https://idp.cloud.vwgroup.com/auth/realms/kums-mfa/protocol/openid-connect/token"
        self.model = "gpt-4o"
        self.base_url = "https://llm.ai.vwgroup.com/v1"
        self.headers = {"X-LLM-API-CLIENT-ID": "IGTF-GmMfMoxTW0VIBhYhU69"}

    def get_token(self) -> str:
        """
        Get the access token for the LLMaaS service.
        """
        response = httpx.post(
            self.url,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": self.grant_type,
            },
        )
        assert response.status_code == 200, f"Error from Cloud IDP: {response.status_code} - {response.text}"
        return response.json()["access_token"]

    def get_model(self):
        """
        Initiates an LLM model from the LLMaaS service.
        """
        token = self.get_token()
        model = langchain_openai.ChatOpenAI(
            model=self.model,
            api_key=token,
            base_url=self.base_url,
            default_headers=self.headers,
        )
        return model

# Create an instance of LLMaaSModel and get the LLM.
llmaas_model_instance = LLMaaSModel()
llm = llmaas_model_instance.get_model()
print(f"Using LLMaaS model: {llm.model_name}")

# -----------------------------------------------------------------------------
# Local in-memory database to manage user accounts.
local_db = {}

# -----------------------------------------------------------------------------
# Account Manager Tools.
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
# Agent descriptions for context.
AGENT_DESCRIPTIONS = {
    "login": "Handles account login by verifying user email and password.",
    "signup": "Handles new account registration by collecting user name, email, and password.",
    "password_reset": "Handles password reset by verifying email and updating the password."
}

# -----------------------------------------------------------------------------
# Define available agent options and system prompt.
members = ["login", "signup", "password_reset"]
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the following workers: "
    "login, signup, and password_reset. Given the user request and the following agent options, "
    "respond with the worker to act next. When finished, respond with FINISH.\n"
    "Agent options:\n"
    "1. login - " + AGENT_DESCRIPTIONS["login"] + "\n" +
    "2. signup - " + AGENT_DESCRIPTIONS["signup"] + "\n" +
    "3. password_reset - " + AGENT_DESCRIPTIONS["password_reset"] + "\n"
)

# -----------------------------------------------------------------------------
# Supervisor node using the custom LLMaaS model.
def supervisor_node(state: dict) -> Command:
    """
    Supervisor node that:
    - Ensures a user request is present in the state.
    - Prepares the system prompt and user messages.
    - Uses the LLMaaS model to decide which agent should handle the request.
    - Manually parses the returned text to route the conversation.
    """
    # If there's no user message, prompt the user.
    if not state.get("messages") or not any(msg["role"] == "user" for msg in state["messages"]):
        user_request = input("What would you like to do? (Options: login, signup, password reset) \n")
        state.setdefault("messages", []).append({"role": "user", "content": user_request})
    
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)
    print(f"[Supervisor] LLM raw response: {response}")
    
    # Extract the text from the response.
    text = response.content if hasattr(response, "content") else str(response)
    text_lower = text.lower()
    
    if "login" in text_lower:
        goto = "login"
    elif "signup" in text_lower:
        goto = "signup"
    elif "password_reset" in text_lower or "reset" in text_lower or "password" in text_lower:
        goto = "password_reset"
    elif "finish" in text_lower:
        goto = END
    else:
        goto = "supervisor"
    
    return Command(goto=goto, update={"next": goto})

# -----------------------------------------------------------------------------
# Agent nodes that execute their specific tool then return control to the supervisor.
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
# Main loop: run the orchestrator until the user decides to exit.
def main():
    print("=== Welcome to the Multi-Agent Account Manager (LangGraph Version) ===")
    while True:
        # Run the graph; the supervisor node now collects a user request if needed.
        for _ in graph.stream({"messages": []}):
            pass
        cont = input("Do you want to perform another action? (yes/no): ")
        if cont.strip().lower() != "yes":
            print("Exiting the system. Goodbye!")
            break

if __name__ == "__main__":
    main()
