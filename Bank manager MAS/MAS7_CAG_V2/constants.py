# constants.py
from langgraph.graph import END

# --- Node Name Constants ---
# Supervisors
L1_SUPERVISOR = "L1_Supervisor"
L2_AUTH = "L2_AuthSupervisor"
L2_ACCOUNT = "L2_AccountSupervisor"
L2_SUPPORT = "L2_SupportSupervisor"

# L3 Tools
L3_LOGIN = "L3_LoginToolNode"
L3_SIGNUP = "L3_SignupToolNode"
L3_PWRESET = "L3_PasswordResetToolNode"
L3_LOGOUT = "L3_LogoutToolNode"
L3_BALANCE = "L3_CheckBalanceToolNode"
L3_HISTORY = "L3_GetHistoryToolNode"
L3_LOGIN_NAME = "L3_LoginNameUpdateToolNode"
L3_HOLDER_NAME = "L3_AccountHolderNameUpdateToolNode"
L3_FAQ = "L3_FAQToolNode"
L3_HANDOFF = "L3_HumanHandoffNode"

# Special Nodes / Actions
CLARIFICATION_NODE = "AskForClarificationNode"
DISPLAY_DETAILS_ACTION = "L2_DISPLAY_DETAILS" # Action handled within L2 Account

# Groupings for convenience
ALL_L2_SUPERVISORS = [L2_AUTH, L2_ACCOUNT, L2_SUPPORT]
ALL_L3_TOOLS = [
    L3_LOGIN, L3_SIGNUP, L3_PWRESET, L3_LOGOUT, L3_BALANCE,
    L3_HISTORY, L3_LOGIN_NAME, L3_HOLDER_NAME, L3_FAQ, L3_HANDOFF
]