# Multi-Level Banking Agent System Flow

This document outlines the typical flow of a user request through the multi-level, multi-agent banking system.

## 1. User Interaction & Initial Input
*   **User:** Submits a request (e.g., "Check my balance", "I want to apply for a loan", "Reset my password") via a chat interface, web portal, or app.

## 2. Level 1: Main Banking Supervisor (Orchestrator)
*   **Receives:** User's raw request.
*   **Action:**
    *   Performs initial **Intent Classification** (e.g., Account Inquiry, Loan Request, Authentication, General Support).
    *   Checks **Authentication Status** (Is the user logged in?).
*   **Decision & Routing:**
    *   **If** request is a simple, common FAQ (e.g., "Bank hours"):
        *   **-> Executes:** `BasicFAQTool`.
        *   **-> Jumps to:** Step 6 (Format & Send Response).
    *   **If** request requires authentication but user is not logged in (or it's an auth-related request like "login", "reset password"):
        *   **-> Routes to:** **Level 2: Auth & Security Supervisor**.
    *   **If** request relates to accounts (and user is authenticated if required):
        *   **-> Routes to:** **Level 2: Account Management Supervisor**.
    *   **If** request relates to transactions/payments (and user is authenticated):
        *   **-> Routes to:** **Level 2: Transaction Supervisor**.
    *   **If** request relates to loans:
        *   **-> Routes to:** **Level 2: Loan Supervisor**.
    *   **If** request relates to investments (and user is authenticated):
        *   **-> Routes to:** **Level 2: Investment Supervisor**.
    *   **If** request is general support, unclear, or cannot be classified:
        *   **-> Routes to:** **Level 2: Customer Support Supervisor**.
    *   **If** more information is needed to classify intent:
        *   **-> Action:** Ask User for Clarification.
        *   **-> Loops back to:** Step 1 (with clarified input).

## 3. Level 2: Departmental Supervisors
*   *(One of the following supervisors becomes active based on L1 routing)*
*   **Receives:** Task delegated from Level 1 Supervisor (e.g., "User wants to check balance", "User wants login help").
*   **Action:**
    *   Refines the specific task needed within its domain.
    *   May check authentication status again if necessary for the specific task.
*   **Decision & Delegation:**
    *   **Identifies** the appropriate **Level 3 Specialist Agent or Tool** for the specific task.
    *   **Delegates** the task:
        *   *Example (Account Supervisor):* If task is "check balance" -> Delegates to `AccountBalanceTool`.
        *   *Example (Auth Supervisor):* If task is "login" -> Delegates to `LoginAgent`.
        *   *Example (Loan Supervisor):* If task is "apply for loan" -> Delegates to `LoanApplicationAgent`.
        *   *(...and so on for other L2 supervisors and their corresponding L3 agents/tools)*
    *   **If** more specific information is needed for the task:
        *   **-> Action:** Ask User for Clarification (via L1).
        *   **-> Loops back to:** Await clarified input.
    *   **If** the request cannot be handled by automated tools/agents within this department (especially Support Supervisor):
        *   **-> Routes to:** Step 7 (Human Agent Handoff).

## 4. Level 3: Specialist Agents & Tools
*   *(One or more L3 agents/tools become active based on L2 delegation)*
*   **Receives:** Specific, atomic task from Level 2 Supervisor (e.g., "Execute login for user X", "Fetch balance for account Y", "Collect loan application data").
*   **Action:**
    *   **Executes** the defined task.
    *   **Interacts with:** **Backend Systems / Databases** (e.g., fetch data, update records, trigger transactions). Requires appropriate API calls and security.
*   **Result:**
    *   Produces a result (e.g., success/failure message, account balance data, transaction confirmation ID, collected data).
*   **Return Path:**
    *   **-> Sends Result back to:** The **Level 2 Supervisor** that delegated the task.

## 5. Return Flow & Processing
*   **Level 2 Supervisor:**
    *   **Receives:** Result from Level 3 Agent/Tool.
    *   **Action:**
        *   Processes the result.
        *   May decide if further L3 steps are needed within its domain (e.g., after collecting application data, trigger a simulation tool).
        *   Formats the result for the main orchestrator.
    *   **Return Path:**
        *   **-> Sends Processed Result back to:** **Level 1 Main Supervisor**.
*   **Level 1 Main Supervisor:**
    *   **Receives:** Processed result from the active Level 2 Supervisor.
    *   **Action:** Prepares the final response for the user. Maintains conversation state.

## 6. Format & Send Response
*   **Level 1 Main Supervisor:**
    *   Formats the final message based on the results received.
*   **System:**
    *   **-> Sends Response:** Delivers the formatted message to the **User** via the interface.
    *   **-> Interaction potentially ends or awaits next user input.**

## 7. Endpoint: Human Agent Handoff (Alternative Flow)
*   **Triggered by:** Level 2 (usually Support) Supervisor or potentially Level 1 if request is explicitly for a human or an unrecoverable error occurs.
*   **Action:**
    *   Collects necessary context (user ID, conversation history).
    *   Initiates transfer to a human agent queue/system.
    *   Informs the user about the handoff.
*   **-> Interaction with automated system ends.**
