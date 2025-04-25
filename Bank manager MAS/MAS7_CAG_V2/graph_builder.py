# graph_builder.py
from langgraph.graph import StateGraph, START, END
import traceback

# Import definitions, nodes, and routers
from state import AppState
from constants import (
    L1_SUPERVISOR, L2_AUTH, L2_ACCOUNT, L2_SUPPORT,
    L3_LOGIN, L3_SIGNUP, L3_PWRESET, L3_LOGOUT, L3_BALANCE,
    L3_HISTORY, L3_LOGIN_NAME, L3_HOLDER_NAME, L3_FAQ, L3_HANDOFF,
    CLARIFICATION_NODE, ALL_L2_SUPERVISORS, ALL_L3_TOOLS
)
from supervisors import (
    l1_main_supervisor_llm, l2_auth_supervisor,
    l2_account_supervisor, l2_support_supervisor
)
from tools import (
    login_tool_node, signup_tool_node, password_reset_tool_node, logout_tool_node,
    check_balance_tool_node, get_history_tool_node, login_name_update_tool_node,
    account_holder_name_update_tool_node, faq_tool_node, human_handoff_node
)
from routing_logic import (
    ask_for_clarification_node, route_l1_decision, route_l2_decision,
    route_after_tool, route_after_clarification
)

print("--- Building Graph ---")

# --- Build the graph ---
builder = StateGraph(AppState)

# Add ALL Nodes
builder.add_node(L1_SUPERVISOR, l1_main_supervisor_llm)
builder.add_node(L2_AUTH, l2_auth_supervisor)
builder.add_node(L2_ACCOUNT, l2_account_supervisor)
builder.add_node(L2_SUPPORT, l2_support_supervisor)
builder.add_node(CLARIFICATION_NODE, ask_for_clarification_node)
builder.add_node(L3_LOGIN, login_tool_node)
builder.add_node(L3_SIGNUP, signup_tool_node)
builder.add_node(L3_PWRESET, password_reset_tool_node)
builder.add_node(L3_LOGOUT, logout_tool_node)
builder.add_node(L3_BALANCE, check_balance_tool_node)
builder.add_node(L3_HISTORY, get_history_tool_node)
builder.add_node(L3_LOGIN_NAME, login_name_update_tool_node)
builder.add_node(L3_HOLDER_NAME, account_holder_name_update_tool_node)
builder.add_node(L3_FAQ, faq_tool_node)
builder.add_node(L3_HANDOFF, human_handoff_node)


# --- Define Edges ---
builder.add_edge(START, L1_SUPERVISOR)

# L1 -> L2 (Unchanged)
builder.add_conditional_edges( L1_SUPERVISOR, route_l1_decision,
    {L2_AUTH: L2_AUTH, L2_ACCOUNT: L2_ACCOUNT, L2_SUPPORT: L2_SUPPORT}
)

# L2 -> L3 / Clarification / L2 / END
l2_targets = {tool: tool for tool in ALL_L3_TOOLS}
l2_targets[CLARIFICATION_NODE] = CLARIFICATION_NODE
l2_targets[L2_AUTH] = L2_AUTH
l2_targets[L2_SUPPORT] = L2_SUPPORT
l2_targets[L3_HANDOFF] = L3_HANDOFF
l2_targets[END] = END
for supervisor_node in ALL_L2_SUPERVISORS:
    builder.add_conditional_edges(supervisor_node, route_l2_decision, l2_targets)

# Edges FROM L3 Nodes -> END / L2_AUTH / L2_SUPPORT
# Using the simpler route_after_tool
after_tool_map = { L2_AUTH: L2_AUTH, L2_SUPPORT: L2_SUPPORT, END: END }
for tool_node in ALL_L3_TOOLS:
     builder.add_conditional_edges(
         tool_node,
         route_after_tool,
         after_tool_map
     )

# Edges FROM Clarification Node -> L3 / L2_SUPPORT / END
clarification_targets = {tool: tool for tool in ALL_L3_TOOLS} # Expects L3 choice
clarification_targets[L2_SUPPORT] = L2_SUPPORT # Fallback on error
clarification_targets[END] = END               # Handle cancellation
builder.add_conditional_edges(
    CLARIFICATION_NODE,
    route_after_clarification,
    clarification_targets
)

# Compile the graph
try:
    graph = builder.compile()
    print("\nGraph compiled successfully (LLM Routing + Clarification)!")
except Exception as e:
    print(f"\nFatal Error: Graph compilation failed: {e}")
    traceback.print_exc()
    exit(1)

# Visualize (Optional)
try:
    output_filename = "banking_agent_graph_cag_clarify.png"
    graph.get_graph().draw_mermaid_png(output_file_path=output_filename)
    print(f"Graph visualization saved to {output_filename}")
except ImportError:
    print("Install pygraphviz and graphviz to visualize the graph: pip install pygraphviz")
except Exception as e:
    print(f"Warning: Could not generate graph visualization: {e}")

print("--- Graph Building Complete ---")