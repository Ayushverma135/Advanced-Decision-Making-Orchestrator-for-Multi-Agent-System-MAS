# state.py
from typing import TypedDict, List, Optional, Sequence, Tuple, Dict, Any
from langchain_core.messages import BaseMessage

# --- State Definition ---
class UserInfo(TypedDict):
    email: str
    name: str
    account_holder_name: str
    account_number: str
    account_id: str
    account_type: str

class AppState(TypedDict):
    messages: Sequence[BaseMessage]
    user_info: Optional[UserInfo]
    current_task: Optional[str] # Task description passed down
    task_result: Optional[str]
    next_action: Optional[str] # Node name to go next
    error_message: Optional[str]
    suggested_choices: Optional[List[Tuple[str, str]]] # For clarification node (name, desc)