Generate by Gemini on Wednesday(01-10-25)
Commit ID - ed70974afcfa23bfb95f5b4f135f12373665bc26

### **Documentation: Doctor Availability Check Feature for MAYA Voice AI**

#### **1. Overview**

This document outlines the recent modifications made to the `app.py` script for the MAYA Voice AI. The primary goal of this update was to move beyond a conversational promise of checking a doctor's availability to a functional, data-driven system.

Previously, MAYA would state its intention to check a doctor's schedule but lacked the backend logic to perform a real-time verification. This update implements a "single source of truth" for doctor availability and equips MAYA with the necessary tools and intelligence to use this data, resulting in more accurate and helpful responses to the caller.

#### **2. Summary of Changes**

The implementation of this feature involved modifications in four key areas of the code:

1.  **Data Structure:** Replaced the simple list of doctor names with a dictionary to store the availability status (available/unavailable) for each doctor.
2.  **AI System Instructions:** Updated MAYA's core prompt to instruct it on *how* and *when* to use the new availability-checking capability.
3.  **Tool Definition:** Enhanced the `check_availability` function tool by adding a `doctor_name` parameter, allowing the AI to pass the specific doctor's name to the backend.
4.  **Backend Logic:** Implemented the core logic within the `_check_availability` function to query the new data structure and return a structured response based on the doctor's status.

#### **3. Detailed Breakdown of Code Changes**

**3.1. Centralized Doctor Availability Data**

The static list of doctors was replaced with a more functional dictionary.

**Before:**

```python
AVAILABLE_DOCTORS = [
    "Dr. Rajesh Kumar",
    "Dr. Priya Sharma", 
    "Dr. Arjun Nair",
    # ... and so on
]
```

**After:**

```python
# --- Doctor Availability Data ---
# This dictionary now controls which doctors are available for appointments.
# True = Available, False = Unavailable.
DOCTOR_AVAILABILITY = {
    "Dr. Rajesh Kumar": True,    # Available
    "Dr. Priya Sharma": False,   # Unavailable
    "Dr. Arjun Nair": True,     # Available
    # ... and so on
}
```

  * **Why this change was made:** A dictionary allows us to associate a state (in this case, a boolean `True` or `False`) with each doctor. This serves as a simple, centralized control panel. To change a doctor's availability, you only need to modify their value in this dictionary.

**3.2. Enhanced AI System Instructions**

Key instructions were added to MAYA's system prompt within the `_get_gemini_config` method to guide its behavior.

**Key additions to `system_instruction`:**

```
- When a user requests an appointment with a specific doctor (e.g., "Dr. Priya Sharma"), you MUST use the `check_availability` function with the doctor's name to verify their availability first.
- Based on the function's result, if the doctor is available, proceed with booking. If the doctor is NOT available, you MUST inform the caller immediately and politely ask if they would like to book with another available doctor. Do NOT say 'let me check' and then wait. Check first, then respond with the result.
- The list of currently available doctors is: {', '.join(available_doctors_list)}.
```

  * **Why this change was made:** These instructions explicitly teach the AI the correct workflow. It now knows it *must* use a tool before making a promise to the user and is told how to handle both "available" and "unavailable" scenarios, ensuring a consistent and honest user experience.

**3.3. Upgraded `check_availability` Tool Definition**

The `FunctionDeclaration` for the `check_availability` tool within `_get_appointment_tools` was updated to accept a doctor's name.

**After:**

```python
FunctionDeclaration(
    name="check_availability",
    description="Check available appointment slots for a specific date and, optionally, a specific doctor.",
    parameters={
        "type": "object",
        "properties": {
            # ... other properties
            "doctor_name": {"type": "string", "description": "The name of the doctor to check for. Example: 'Dr. Rajesh Kumar'"},
        },
        "required": ["date"]
    }
),
```

  * **Why this change was made:** This modification exposes the `doctor_name` parameter to the AI. It allows the language model to extract the doctor's name from the user's speech and pass it as a structured argument to the Python backend for processing.

**3.4. Implemented Backend Logic for Availability Check**

The `_check_availability` async function was rewritten to perform the actual lookup.

**Key logic of the new function:**

1.  It retrieves the `doctor_name` from the arguments passed by the AI.
2.  If a name is provided, it looks up that name in the global `DOCTOR_AVAILABILITY` dictionary.
3.  **Scenario 1: Doctor is Unavailable** (`False`): It returns a JSON object with `status: "doctor_unavailable"`.
4.  **Scenario 2: Doctor is Not Found:** It returns a `status: "doctor_not_found"` to handle cases of misheard names.
5.  **Scenario 3: Doctor is Available (`True`) or No Doctor Specified:** It proceeds to return the standard list of available time slots with `status: "slots_available"`.

<!-- end list -->

  * **Why this change was made:** This is the core engine of the new feature. It connects the AI's request to the data source and provides a clear, structured response that the AI can easily interpret to formulate its verbal reply.

-----

#### **4. How the New Feature Works: A Step-by-Step Workflow**

Here is the end-to-end process of how MAYA now handles a request for a specific doctor:

1.  **User Request:** A caller asks, *"Dr. Priya Sharma-yude appointment kittumo?"* (Can I get an appointment with Dr. Priya Sharma?).
2.  **Intent Recognition:** MAYA's language model processes the audio, understands the intent is to book an appointment, and identifies the entity "Dr. Priya Sharma".
3.  **Tool Selection:** Guided by its new system instructions, MAYA knows it must first verify the doctor's availability. It calls the `check_availability` tool, passing `doctor_name="Dr. Priya Sharma"` as an argument.
4.  **Backend Execution:** The `_check_availability` function in Python is triggered. It looks up "Dr. Priya Sharma" in the `DOCTOR_AVAILABILITY` dictionary and finds the value is `False`.
5.  **Structured Response:** The function returns a JSON object to the AI, for example:
    ```json
    {
        "status": "doctor_unavailable",
        "doctor_name": "Dr. Priya Sharma",
        "available_slots": []
    }
    ```
6.  **AI Response Generation:** MAYA receives this structured data. It interprets `status: "doctor_unavailable"` and, following its instructions, generates a polite, helpful response in Malayalam.
7.  **Final Output:** MAYA speaks to the user: *"Kshamikkanam, Dr. Priya Sharma inn available alla. Mattoru doctor-ude appointment edukkano?"* (I'm sorry, Dr. Priya Sharma is not available today. Would you like to book an appointment with another doctor?).