# descriptions_new.py
from typing import List, Dict, Any

# --- Agent Route Definitions ---
agent_route_definitions: List[Dict[str, Any]] = [
    {
        "name": "LoanAgent",
        "description": "Handles user queries related to loans, eligibility, application, and loan status.",
        "keywords": ["loan", "loans", "borrow", "credit", "mortgage", "finance", "eligibility", "apply loan", "loan status", "loan calculator"]
    },
    {
        "name": "TravelingAgent",
        "description": "Assists users with travel planning, including flights, hotels, car rentals, and itineraries.",
        "keywords": ["travel", "trip", "flight", "flights", "hotel", "hotels", "car rental", "rent a car", "itinerary", "vacation", "holiday"]
    },
    {
        "name": "ServiceRequestAgent",
        "description": "Manages service requests, scheduling, and checking provider availability for various services.",
        "keywords": ["service", "services", "request service", "schedule service", "book service", "provider availability", "service type"]
    }
]

# --- Loan Agent Tool Definitions ---
loan_tool_definitions: List[Dict[str, Any]] = [
    {
        "name": "LoanApplyTool",
        "description": "Guides user through the loan application process.",
        "keywords": ["apply for loan", "loan application", "submit application", "loan request"]
    },
    {
        "name": "LoanEligibilityTool",
        "description": "Checks user's eligibility for a loan.",
        "keywords": ["eligibility", "eligible", "qualify", "requirements", "criteria"]
    },
    {
        "name": "LoanTypeTool",
        "description": "Helps user identify the right type of loan (student, home, personal).",
        "keywords": ["loan type", "type of loan", "student loan", "home loan", "personal loan"]
    },
    {
        "name": "LoanStatusTool",
        "description": "Checks the status of a loan application.",
        "keywords": ["loan status", "application status", "check loan status", "track application"]
    },
    {
        "name": "LoanCalculatorTool",
        "description": "Calculates loan payments.",
        "keywords": ["loan calculator", "payment calculation", "calculate loan", "loan amount"]
    }
]

# --- Traveling Agent Tool Definitions ---
traveling_tool_definitions: List[Dict[str, Any]] = [
    {
        "name": "FlightSearchTool",
        "description": "Searches for flights.",
        "keywords": ["flight search", "find flights", "flights to", "book flight"]
    },
    {
        "name": "HotelSearchTool",
        "description": "Searches for hotels.",
        "keywords": ["hotel search", "find hotels", "hotels in", "book hotel"]
    },
    {
        "name": "CarRentalTool",
        "description": "Searches for car rentals.",
        "keywords": ["car rental", "rent a car", "car hire", "rental car"]
    },
    {
        "name": "ItineraryPlanningTool",
        "description": "Helps plan a travel itinerary.",
        "keywords": ["itinerary planning", "plan itinerary", "travel plan", "trip itinerary"]
    },
    {
        "name": "PaymentProcessingTool",
        "description": "Processes payments for travel bookings.",
        "keywords": ["payment processing", "make payment", "pay for booking", "secure payment"]
    },
    {
        "name": "TravelNotificationTool",
        "description": "Sends travel notifications (flight updates, etc.).",
        "keywords": ["travel notifications", "flight alerts", "trip reminders", "updates"]
    },
    {
        "name": "WeatherForecastTool",
        "description": "Provides weather forecasts for travel destinations.",
        "keywords": ["weather forecast", "weather in", "forecast for", "weather prediction"]
    },
    {
        "name": "LocalInsightsTool",
        "description": "Provides local insights and recommendations for travel destinations.",
        "keywords": ["local insights", "local tips", "things to do in", "recommendations"]
    }
]

# --- Service Request Agent Tool Definitions ---
service_request_tool_definitions: List[Dict[str, Any]] = [
    {
        "name": "ServiceTypeTool",
        "description": "Helps user identify the type of service needed.",
        "keywords": ["service type", "type of service", "select service", "choose service"]
    },
    {
        "name": "SchedulingServiceDateTool",
        "description": "Schedules a date and time for the service.",
        "keywords": ["schedule service", "book appointment", "set date", "arrange time"]
    },
    {
        "name": "ServiceProviderAvailabilityCheckTool",
        "description": "Checks service provider availability.",
        "keywords": ["provider availability", "check availability", "service provider available", "when can you come"]
    }
]

# --- Create global lookup for descriptions ---
all_definitions_list = (agent_route_definitions + loan_tool_definitions +
                       traveling_tool_definitions + service_request_tool_definitions)
description_lookup: Dict[str, str] = {d["name"]: d["description"] for d in all_definitions_list}

def get_node_description(node_name: str) -> str:
    """Looks up a brief description for a given node name."""
    return description_lookup.get(node_name, "Perform this action") # Default description