# main_new.py
from typing import Dict, Any
from langchain_core.messages import HumanMessage

from agents_new import main_supervisor_agent, planner_agent, loan_agent, traveling_agent, service_request_agent # Import specialized agents too
from tools_new import * # Import all tools


# Define AppState - simplified for this example
AppState = Dict[str, Any]

# Tool Mapping - connect tool names from Planner to tool functions
TOOL_MAPPING = {
    "LoanEligibilityTool": loan_eligibility_tool,
    "LoanTypeTool": loan_type_tool,
    "LoanApplyTool": loan_apply_tool,
    "LoanStatusTool": loan_status_tool,
    "LoanCalculatorTool": loan_calculator_tool,
    "FlightSearchTool": flight_search_tool,
    "HotelSearchTool": hotel_search_tool,
    "CarRentalTool": car_rental_tool,
    "ItineraryPlanningTool": itinerary_planning_tool,
    "PaymentProcessingTool": payment_processing_tool,
    "TravelNotificationTool": travel_notification_tool,
    "WeatherForecastTool": weather_forecast_tool,
    "LocalInsightsTool": local_insights_tool,
    "ServiceTypeTool": service_type_tool,
    "SchedulingServiceDateTool": scheduling_service_date_tool,
    "ServiceProviderAvailabilityCheckTool": service_provider_availability_check_tool,
}


def main():
    print("\n=== Welcome to the Multi-Agent System (New Domains - Planner LLM + Clarification) ===")
    print("You can ask about loans, travel, or service requests.")
    print("Type 'quit' or 'exit' to end the conversation.")

    current_state: AppState = {
        "messages": [],
        "current_task": None,
        "task_result": None,
        "next_action": None,
        "error_message": None,
        "tool_results": {}, # To store results of each tool execution
        "agent_options": None, # To store multiple agent options from Supervisor
    }

    while True:
        final_task_result_prev_turn = current_state.get("task_result")
        if final_task_result_prev_turn:
            print(f"\nAssistant: {final_task_result_prev_turn}")

        user_input = input("\nYou: ")

        if 'quit' in user_input.lower() or 'exit' in user_input.lower():
            print("Assistant: Goodbye!")
            break

        current_messages = current_state.get('messages', [])
        if not isinstance(current_messages, list):
             current_messages = list(current_messages)
        current_messages.append(HumanMessage(content=user_input))

        current_state = {
            "messages": current_messages,
            "current_task": user_input, # Store user input as current task
            "task_result": None,
            "next_action": None,
            "error_message": None,
            "tool_results": {}, # Reset tool results for new turn
            "agent_options": None, # Clear agent options for new turn
        }

        print("\nAssistant Processing...")

        try:
            # 1. Main Supervisor Agent
            supervisor_result = main_supervisor_agent(user_input) # Now can return agent name OR list of options

            if isinstance(supervisor_result, list): # Multiple agent options returned
                current_state["agent_options"] = supervisor_result
                print("\n--- Multiple Agent Options Found ---")
                print("I'm not sure which agent you need. Did you mean:")
                for i, option in enumerate(supervisor_result):
                    print(f"  {i+1}. {option['name']} (Relevance Score: {option['score']:.2f})")

                while True: # Loop until valid choice or exit
                    choice_str = input(f"Please enter the number of your choice (1-{len(supervisor_result)}), or type 'exit' to cancel: ")
                    if choice_str.lower() == 'exit':
                        current_state["task_result"] = "Clarification cancelled."
                        break # Exit clarification loop, back to main loop
                    try:
                        choice_index = int(choice_str) - 1
                        if 0 <= choice_index < len(supervisor_result):
                            selected_agent_name = supervisor_result[choice_index]["name"]
                            print(f"User chose agent: {selected_agent_name}")
                            break # Valid choice made, exit clarification loop
                        else:
                            print(f"Invalid choice. Please enter a number between 1 and {len(supervisor_result)} or 'exit'.")
                    except ValueError:
                        print("Invalid input. Please enter a number or 'exit'.")

                if current_state["task_result"] == "Clarification cancelled.":
                    continue # Skip to next user input prompt if cancelled

            elif isinstance(supervisor_result, str): # Single agent name returned (or "UnknownAgent")
                selected_agent_name = supervisor_result
                print(f"\n--- Main Agent Selected: {selected_agent_name} ---")

                if selected_agent_name == "UnknownAgent":
                    current_state["task_result"] = "Sorry, I couldn't understand which agent can handle your request. Please be more specific."
                    print(f"\nAssistant Response: {current_state['task_result']}")
                    continue # Skip planner and tool execution, go to next user input

            else: # Should not happen, but handle for robustness
                selected_agent_name = "UnknownAgent"
                current_state["error_message"] = "Unexpected response from Main Supervisor Agent."
                current_state["task_result"] = "Sorry, there was an internal error in routing your request."
                print(f"\nAssistant Response: {current_state['task_result']}")
                continue # Skip planner and tool execution, go to next user input


            # 2. Planner Agent (LLM for Tool Sequencing) - Proceed only if a valid agent is selected
            if selected_agent_name != "UnknownAgent":
                planner_response = planner_agent(user_input, selected_agent_name)
                tool_sequence = planner_response.get("tool_sequence", [])

                print(f"\n--- Executing Tool Sequence: {tool_sequence} ---")

                tool_results = {} # Store results of each tool

                # 3. Tool Execution Sequence
                for tool_name in tool_sequence:
                    if tool_name in TOOL_MAPPING:
                        tool_function = TOOL_MAPPING[tool_name]
                        print(f"\n--- Executing Tool: {tool_name} ---")
                        tool_output = tool_function(user_input) # Pass user_input for context
                        tool_results[tool_name] = tool_output # Store individual tool results
                        print(f"--- Tool '{tool_name}' Result: {tool_output.get('tool_result')}") # Print tool result
                    else:
                        error_message = f"Error: Tool '{tool_name}' not found in TOOL_MAPPING."
                        print(error_message)
                        current_state["error_message"] = error_message
                        tool_results[tool_name] = {"tool_result": error_message} # Store error result

                current_state["tool_results"] = tool_results # Store all tool results in state

                # 4. Response to User (Basic response for now - can be improved)
                if tool_sequence:
                    final_response = f"Executed tool sequence: {', '.join(tool_sequence)}. Check individual tool results." # Basic summary
                elif planner_response and planner_response.get("agent_response"):
                    final_response = planner_response["agent_response"] # In case Planner agent has a specific message
                else:
                    final_response = "Sorry, I could not process your request."

                current_state["task_result"] = final_response
                print(f"\nAssistant Response: {final_response}")


        except Exception as e:
            print(f"\n--- Error during processing: {e} ---")
            current_state["task_result"] = "Sorry, there was an error processing your request."
            print(f"\nAssistant Response: {current_state['task_result']}")


if __name__ == "__main__":
    main()