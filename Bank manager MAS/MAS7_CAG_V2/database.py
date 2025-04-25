# database.py
import random
import string
import uuid
from typing import Dict, Any

# --- Helper functions ---
def generate_account_number() -> str:
    return ''.join(random.choices(string.digits, k=10))

def generate_account_id() -> str:
    return str(uuid.uuid4())

# --- Databases ---
local_db: Dict[str, Any] = {
    "ayush@gmail.com": {
        "name": "ayush135", "account_holder_name": "Ayush Sharma", "password": "123",
        "balance": 1500.75, "history": ["+ $1000 (Initial Deposit)", "- $50 (Groceries)", "+ $600.75 (Salary)"],
        "account_number": generate_account_number(),
        "account_id": generate_account_id(), "account_type": "Savings"
    }
}

faq_db: Dict[str, str] = {
    "hours": "Our bank branches are open Mon-Fri 9 AM to 5 PM. Online banking is available 24/7.",
    "contact": "You can call us at 1-800-BANKING or visit our website's contact page.",
    "locations": "We have branches in Pune and Gurugram. Use our online locator for specific addresses."
}