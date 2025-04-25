# main.py
import asyncio
import traceback
from typing import Dict, Any

# Import state, message types, graph
from state import AppState # Assuming AppState is correctly defined here or imported
from langchain_core.messages import HumanMessage
from graph_builder import graph # Import the compiled graph
from langgraph.graph import END # Import END state
import time # Keep time import if used elsewhere (e.g., handoff simulation)

async def main_async():
    """Asynchronous main execution loop."""
    # Use the welcome message matching the reference snippet
    print("\n=== Welcome to the Multi-Level Banking Assistant (v9 - LLM Routing + Clarification) ===")
    print("You can ask about balance, history, FAQs, login, signup, password reset, logout, name updates.")
    print("Type 'quit' or 'exit' to end the conversation.")

    # Initial state matching the reference snippet
    current_state: AppState = {
        "messages": [],
        "user_info": None,
        "current_task": None,
        "task_result": None,
        "next_action": None,
        "error_message": None,
        "suggested_choices": None,
        # top_matches_from_l2 is confirmed removed
    }

    turn_counter = 0 # Keep for debugging if needed

    while True:
        turn_counter += 1
        print(f"\n--- START OF TURN {turn_counter} ---")
        # Using .get() for safety, although it should exist
        print(f"State BEFORE Input: {current_state}")

        # --- Print results/errors from previous turn ---
        # This logic seems correct and matches the reference behavior
        if current_state.get("task_result"):
            print(f"\nAssistant: {current_state['task_result']}")
        if current_state.get("error_message"):
             error_msg = current_state['error_message']
             # ... (error message formatting copied from reference) ...
             if "Ambiguous request" in error_msg: print(f"\nAssistant: I'm not sure exactly what you meant. {error_msg}. Could you please clarify or ask for 'help'?")
             elif "LLM parsing error" in error_msg or "LLM Error" in error_msg: print(f"\nAssistant: Sorry, I had trouble understanding the options. Please try rephrasing or ask for 'help'. ({error_msg})")
             elif error_msg == "Authentication Required": print(f"\nAssistant: Please log in first to complete your request.")
             elif error_msg in ["Authentication Failed", "Account Not Found", "Email Exists"]: print(f"\nAssistant: There was an issue with authentication: {error_msg}. Please try again.")
             elif "Invalid" in error_msg or "Missing" in error_msg: print(f"\nAssistant: There was an input error: {error_msg}. Please try again.")
             elif "Internal error" in error_msg or "Routing Decision" in error_msg: print(f"\nAssistant: Sorry, an internal routing error occurred. Please try again or ask for 'help'. ({error_msg})")
             else: print(f"\nAssistant: Sorry, I encountered an issue: {error_msg}. Please try asking differently or ask for 'help'.")


        # --- Get User Input ---
        # This logic seems correct and matches the reference behavior
        user_info_current = current_state.get("user_info")
        login_display_name = user_info_current.get("name") if user_info_current else None
        auth_display = f"(Logged in as: {login_display_name})" if login_display_name else "(Not Logged In)"
        user_input = input(f"\nYou {auth_display}: ")

        if 'quit' in user_input.lower() or 'exit' in user_input.lower():
            print("Banking Assistant: Goodbye!")
            break

        # --- Prepare state for the new turn (matches reference snippet) ---
        messages_current_turn = list(current_state.get('messages', []))
        messages_current_turn.append(HumanMessage(content=user_input))

        # Reset turn-specific state, keep user_info and messages
        # This is the state passed *into* the graph stream
        input_state_for_graph: AppState = {
            "messages": messages_current_turn,
            "user_info": current_state.get("user_info"), # Preserve login status
            "current_task": None,
            "task_result": None,
            "next_action": None,
            "error_message": None,
            "suggested_choices": None,
        }
        print(f"State BEFORE Graph Stream (Turn {turn_counter}): {input_state_for_graph}")


        print("\nAssistant Processing...")
        try:
            # --- State Update Logic matching the reference snippet ---
            # This captures the output delta of the last node, not the fully merged state
            final_state_update = None # Initialize to None
            async for event in graph.astream(input_state_for_graph, {"recursion_limit": 25}):
                node_name = list(event.keys())[0]
                node_output_delta = event[node_name] # Get the dictionary returned by the node
                if "_node" in node_name or "Supervisor" in node_name or node_name == END:
                     # Print the delta returned by the node
                     print(f"--- Event (Turn {turn_counter}): Node '{node_name}' Output DELTA: {node_output_delta} ---")
                # Keep track of the *last delta* received
                final_state_update = node_output_delta

            # --- Update current_state based on the *last delta* ---
            # This was the problematic logic, but matches the reference snippet provided
            if final_state_update:
                 print(f"Final Update from Stream (Turn {turn_counter}): {final_state_update}")
                 # Apply the last update delta to the state that was INPUT to the graph
                 # Note: This assumes the delta contains all necessary changes for the *next* turn's state.
                 # It might overwrite fields incorrectly if the delta doesn't include everything needed.
                 current_state = input_state_for_graph.copy() # Start from input state for the turn
                 current_state.update(final_state_update) # Apply only the last delta
                 print(f"--- End of Turn {turn_counter} State (Updated with Last Delta): UserInfo={current_state.get('user_info')} ---")
            else:
                print(f"Warning (Turn {turn_counter}): Graph stream finished empty. State likely preserved from input.")
                # If stream was empty, the state remains the input_state_for_graph,
                # but we should clear transient fields for the next loop iteration
                current_state = input_state_for_graph.copy()
                current_state["task_result"] = None
                current_state["next_action"] = None
                current_state["error_message"] = None
                current_state["suggested_choices"] = None
                current_state["current_task"] = None # Task was for the turn that yielded nothing


            # --- Post-Turn Check (using the potentially incorrect state) ---
            if (current_state.get("next_action") != END and
                not current_state.get("task_result") and
                not current_state.get("error_message") and
                not current_state.get("suggested_choices")):
                 print("\nAssistant: How else can I assist you today? (Idle)")


        except Exception as e:
            # ... (critical error handling unchanged) ...
            print(f"\n--- Critical Error during graph execution (Turn {turn_counter}) ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            traceback.print_exc()
            print("\nAssistant: Encountered a critical system error.")
            break

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    # ... (exception handling unchanged) ...
    except KeyboardInterrupt:
        print("\nExiting...")
    except RuntimeError as e:
        if "Cannot run the event loop while another loop is running" in str(e):
            print("Detected running event loop. Attempting to run main_async differently.")
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main_async())
        else:
            print(f"Runtime Error: {e}")