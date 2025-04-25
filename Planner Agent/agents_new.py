# agents_new.py
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
import json # For parsing LLM tool sequence

# Assuming you have LLM setup (replace with your actual LLM)
from LLMaas import LLMaaSModel
llmaas_model_instance = LLMaaSModel()
llm = llmaas_model_instance.get_model()

from descriptions_new import (
    agent_route_definitions, loan_tool_definitions, traveling_tool_definitions,
    service_request_tool_definitions, get_node_description
)
from tools_new import * # Import all tools

# --- Helper function to format descriptions for prompts ---
def format_descriptions_for_prompt(definitions: List[Dict]) -> str:
    return "\n".join([f"- {d['name']}: {d['description']}" for d in definitions])

# Pre-format agent descriptions for Main Supervisor prompt
AGENT_PROMPT_CONTEXT = format_descriptions_for_prompt(agent_route_definitions)

# --- Main Supervisor Agent (LLM Based) - Unchanged ---

MAIN_SUPERVISOR_SYSTEM_PROMPT = """You are the main routing supervisor for a multi-agent system.
Your goal is to determine the relevant agent(s) to handle the user's request and their relevance scores.

Available Agents:
{agent_descriptions}

Rules:
- Analyze the user's request: "{user_request}"
- Identify ALL agents from the list above that could potentially handle the request.
- For each potentially relevant agent, assign a relevance score between 0.0 (not relevant) and 1.0 (highly relevant). A score >= 0.2 indicates relevance.
- If more than two agents are relevant (score >= 0.2), return the top three options with their names and scores in a JSON list.
- If exactly two agents are relevant (score >= 0.2), return those two options with their names and scores in a JSON list.
- If only one agent is relevant (score >= 0.2), return that one agent's name and score in a JSON list.
- If no agents are relevant (score < 0.2), respond with an empty JSON list: [].
- **Crucially:** Respond ONLY with a valid JSON list of dictionaries. Each dictionary must have keys "name" (the agent's name) and "score" (a float between 0.0 and 1.0).
- Only include agents with a relevance score >= 0.2 in the list.

Example Output (for a request like 'loan and travel'):
[
  {{"name": "LoanAgent", "score": 0.85}},
  {{"name": "TravelingAgent", "score": 0.75}}
]

Example Output (if no agents match):
[]
"""

def main_supervisor_agent(user_query: str) -> str: # Return type will be different now, but keep str for now for simplicity
    """Routes user requests to specialized agents using LLM, now with scoring and code fence handling."""
    print("\n--- Main Supervisor Agent (LLM Based - Scoring & Code Fence Handling) ---")
    print(f"User Query: '{user_query}'")

    system_prompt = MAIN_SUPERVISOR_SYSTEM_PROMPT.format(agent_descriptions=AGENT_PROMPT_CONTEXT, user_request=user_query)
    prompt_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]

    agent_options_with_scores = [] # To store agent options with scores
    try:
        response = llm.invoke(prompt_messages)
        llm_output_str = response.content.strip()
        print(f"Main Supervisor LLM Decision Raw: {llm_output_str}")

        # --- Handle potential markdown code fences ---
        if llm_output_str.startswith("```json"):
            llm_output_str = llm_output_str[7:] # Remove ```json
            if llm_output_str.endswith("```"):
                llm_output_str = llm_output_str[:-3] # Remove ```
            llm_output_str = llm_output_str.strip() # Remove any extra whitespace

        try:
            parsed_output = json.loads(llm_output_str)
            if isinstance(parsed_output, list):
                valid_agents = [agent["name"] for agent in agent_route_definitions]
                for item in parsed_output:
                    if (isinstance(item, dict) and
                        "name" in item and "score" in item and
                        item["name"] in valid_agents and
                        isinstance(item["score"], (int, float)) and
                        0.0 <= item["score"] <= 1.0 and item["score"] >= 0.2): # Ensure score is >= 0.2
                        agent_options_with_scores.append(item)
                    else:
                        print(f"Warning: Invalid item format in LLM output: {item}")
            else:
                print(f"Warning: LLM output is not a JSON list: {llm_output_str}")


        except json.JSONDecodeError as e:
            print(f"Error parsing Main Supervisor LLM JSON output: {e}. Raw output: {llm_output_str}")
            agent_options_with_scores = [] # Treat parse error as no valid agents

    except Exception as e:
        print(f"Error in Main Supervisor LLM call: {e}")
        agent_options_with_scores = [] # Treat LLM call error as no valid agents


    print(f"Main Supervisor Agent Options with Scores: {agent_options_with_scores}")

    if not agent_options_with_scores:
        return "UnknownAgent" # No relevant agents
    elif len(agent_options_with_scores) == 1:
        return agent_options_with_scores[0]["name"] # Single best agent - return name directly
    else:
        # For now, just return the name of the top agent (you can modify this logic later)
        # top_agent_name = agent_options_with_scores[0]["name"]
        # print(f"Multiple agents relevant. For now, routing to top agent: {top_agent_name}")
        # return top_agent_name # Returning just the top agent name for now, but you have options with scores available

        # --- Now returning the list of agent options so main_new.py can handle clarification ---
        return agent_options_with_scores # Return list of agent options for clarification

        # --- Potential future logic for handling multiple options (in main_new.py) ---
        # - Clarification to user: "Did you mean Loan or Travel?"
        # - Parallel execution of agents
        # - More complex routing based on scores


# --- Planner Agent (LLM Based for Tool Sequencing) ---
PLANNER_AGENT_SYSTEM_PROMPT = """You are the Planner Agent.
Your goal is to create a sequence of tools to execute to fulfill the user's request, within the context of the selected agent.

Selected Agent: {selected_agent_name}
User Request: {user_query}

Available Tools for {selected_agent_name}:
{agent_tool_descriptions}

Rules:
- Analyze the user request and the available tools for the selected agent.
- Determine a logical sequence of tools to execute to best address the user's request.
- Respond ONLY with a JSON list of tool names in the desired execution order.
- If no tools are relevant or a sequence cannot be determined, respond with an empty JSON list: [].

Example Output (for a travel request like 'I want to travel from London to Paris next week'):
["ItineraryPlanningTool", "FlightSearchTool", "HotelSearchTool"]
"""

def planner_agent(user_query: str, selected_agent: str) -> Dict[str, Any]:
    """Plans and executes tools based on the selected agent using LLM for sequencing."""
    print("\n--- Planner Agent ---")
    print(f"Selected Agent: {selected_agent}, User Query: '{user_query}'")

    if selected_agent == "LoanAgent":
        tool_definitions = loan_tool_definitions
    elif selected_agent == "TravelingAgent":
        tool_definitions = traveling_tool_definitions
    elif selected_agent == "ServiceRequestAgent":
        tool_definitions = service_request_tool_definitions
    else:
        return {"agent_response": f"Planner Agent received unknown agent: {selected_agent}", "tool_sequence": [], "tool_results": {}}

    agent_tool_prompt_context = format_descriptions_for_prompt(tool_definitions)

    system_prompt = PLANNER_AGENT_SYSTEM_PROMPT.format(
        selected_agent_name=selected_agent,
        user_query=user_query,
        agent_tool_descriptions=agent_tool_prompt_context
    )
    prompt_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]

    tool_sequence = []
    try:
        response = llm.invoke(prompt_messages)
        llm_output_str = response.content.strip()
        print(f"Planner Agent LLM Decision Raw: {llm_output_str}")

        try:
            # Parse LLM output as JSON list of tool names
            parsed_output = json.loads(llm_output_str)
            if isinstance(parsed_output, list):
                tool_sequence = [tool_name for tool_name in parsed_output if tool_name in [tool['name'] for tool in tool_definitions]]
            else:
                print(f"Warning: Planner Agent LLM output is not a JSON list: {llm_output_str}")
        except json.JSONDecodeError as e:
            print(f"Error parsing Planner Agent LLM JSON output: {e}. Raw output: {llm_output_str}")

    except Exception as e:
        print(f"Error in Planner Agent LLM call: {e}")

    print(f"Planner Agent Tool Sequence: {tool_sequence}")
    return {"agent_response": f"Planner Agent determined tool sequence for {selected_agent}.", "tool_sequence": tool_sequence, "tool_results": {}}


# --- Specialized Agents (Simplified - now just tool lists) ---
# Specialized Agents are now primarily used to determine which agent's tools are available for the Planner

def loan_agent(user_query: str) -> Dict[str, Any]: # Simplified, user_query not really used directly here anymore
    """Handles loan-related queries (now just provides tool list to Planner)."""
    print("\n--- Loan Agent (Simplified) ---") # Now primarily used for tool list
    return {"agent_response": "Loan Agent available.", "tools": loan_tool_definitions}

def traveling_agent(user_query: str) -> Dict[str, Any]: # Simplified
    """Handles travel-related queries (now just provides tool list to Planner)."""
    print("\n--- Traveling Agent (Simplified) ---") # Now primarily used for tool list
    return {"agent_response": "Traveling Agent available.", "tools": traveling_tool_definitions}


def service_request_agent(user_query: str) -> Dict[str, Any]: # Simplified
    """Handles service request queries (now just provides tool list to Planner)."""
    print("\n--- Service Request Agent (Simplified) ---") # Now primarily used for tool list
    return {"agent_response": "Service Request Agent available.", "tools": service_request_tool_definitions}