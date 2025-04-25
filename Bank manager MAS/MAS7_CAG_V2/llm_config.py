# llm_config.py
import traceback
from LLMaas import LLMaaSModel # Your original import
# from langchain.globals import set_llm_cache
# from langchain.cache import InMemoryCache # Or SQLiteCache etc.

print("--- Initializing LLM and Cache ---")

# Initialize Caching (do this BEFORE initializing the LLM)
# Using simple in-memory cache for demonstration
# print("Setting up LLM cache (InMemoryCache)...")
# set_llm_cache(InMemoryCache())
# print("LLM Cache enabled.")

# Initialize LLM
llm = None
try:
    llmaas_model_instance = LLMaaSModel()
    llm = llmaas_model_instance.get_model()
    print(f"Using LLM from LLMaaS: {llm.model_name if hasattr(llm, 'model_name') else type(llm)}")
except ImportError as e:
     print(f"Error importing Langchain LLM: {e}. Please install required packages (e.g., pip install langchain-openai).")
     exit(1)
except Exception as e:
    print(f"Fatal Error: Could not initialize LLM model: {e}")
    traceback.print_exc()
    exit(1)

if llm is None:
    print("FATAL: LLM initialization failed.")
    exit(1)

print("--- LLM and Cache Setup Complete ---")