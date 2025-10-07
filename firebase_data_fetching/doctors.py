import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import json

# --- Helper Functions (No changes needed here) ---

def _format_time_from_float(time_float):
    """Converts a time float (e.g., 18.5) to a 24-hour string format (e.g., "18:30")."""
    if time_float is None:
        return "00:00"
    try:
        hours = int(time_float)
        minutes = int((time_float - hours) * 60)
        return f"{hours:02d}:{minutes:02d}"
    except (ValueError, TypeError):
        return "00:00"

def _format_availability_schedule(consultation_times):
    """
    Formats the raw consultationTimes map from Firestore into a structured dictionary.
    """
    schedule = {}
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    if not consultation_times:
        return {day: [] for day in days_order}

    for day in days_order:
        day_schedule = consultation_times.get(day)
        schedule[day] = []
        
        if day_schedule and isinstance(day_schedule, dict):
            for session_data in day_schedule.values():
                if session_data and isinstance(session_data, dict):
                    schedule[day].append({
                        "from": _format_time_from_float(session_data.get('from')),
                        "to": _format_time_from_float(session_data.get('to')),
                        "slots": session_data.get('tokenLimit', 0)
                    })
    
    return schedule

# --- MODIFIED Main Data Fetching Function ---

def fetch_all_doctor_details(clinic_id):
    """
    Fetches the clinic name and complete details for all its doctors using the clinic ID.

    Args:
        clinic_id (str): The document ID of the clinic (e.g., 'QV070').

    Returns:
        dict: A dictionary containing the clinic_name and a nested 'doctors' dictionary.
              e.g., {"clinic_name": "Jaza Healthcare", "doctors": {...}}
    """
    doctor_details_dict = {}
    clinic_name = "the hospital"

    try:
        # 1. Initialize Firebase with the correct credentials file
        cred = credentials.Certificate("firebase_data_fetching/docbooking.json")
        
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        print("Successfully connected to Firestore.")
        print(f"Fetching details for clinic ID: '{clinic_id}'...")

        # 2. Directly get the clinic document by its ID
        clinic_ref = db.collection('clinics').document(clinic_id)
        clinic_doc = clinic_ref.get()

        if not clinic_doc.exists:
            print(f"Error: No clinic found with ID '{clinic_id}'.")
            return {}

        # 3. Fetch the clinic's name from the document's 'name' field
        clinic_data = clinic_doc.to_dict()
        clinic_name = clinic_data.get('name', clinic_name)
        print(f"Found clinic: '{clinic_name}'. Fetching doctor details...")
        
        # 4. Access the 'doctors' subcollection
        doctors_ref = clinic_doc.reference.collection('doctors')
        doctor_docs = doctors_ref.stream()

        # 5. Iterate through each doctor and build the detailed dictionary
        today_str = datetime.now().strftime('%A')

        for doc in doctor_docs:
            doctor_data = doc.to_dict()
            doctor_name = doctor_data.get('name')

            if not doctor_name:
                continue

            availability_schedule = _format_availability_schedule(doctor_data.get('consultationTimes'))
            is_available_today = bool(availability_schedule.get(today_str))

            doctor_details_dict[doctor_name] = {
                "specialization": doctor_data.get('specialization', 'N/A'),
                "consultation_fee": doctor_data.get('consultationFees', 0),
                "is_available_today": is_available_today,
                "availability": availability_schedule
            }
        
        if not doctor_details_dict:
            print("Clinic found, but no doctors were located in its subcollection.")

    except FileNotFoundError:
        print("Error: 'firebase_data_fetching/docbooking.json' not found.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}
    
    print(f"Successfully fetched details for {len(doctor_details_dict)} doctors.")
    # 6. Return the structured dictionary with clinic name and doctor details
    return {
        "clinic_name": clinic_name,
        "doctors": doctor_details_dict
    }

# --- MODIFIED Main Execution (for testing) ---
if __name__ == "__main__":
    # Use the Clinic ID directly for fetching
    TARGET_CLINIC_ID = "QV070" 
    
    clinic_data = fetch_all_doctor_details(TARGET_CLINIC_ID)

    print("\n--- Final Fetched Data Structure ---")
    if clinic_data:
        print(json.dumps(clinic_data, indent=2))
    else:
        print("No data was fetched.")