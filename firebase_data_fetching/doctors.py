import firebase_admin
from firebase_admin import credentials, firestore
import json

def fetch_clinic_data(clinic_id):
    """
    Fetches comprehensive data for a given clinic ID and organizes it into
    four distinct dictionaries.

    Args:
        clinic_id (str): The document ID of the clinic (e.g., 'QV070').

    Returns:
        tuple: A tuple containing four data structures:
               - department_wise_doctors (dict): Doctors grouped by department.
               - doctor_consultation_fee (dict): Fees for each doctor.
               - doctor_availability (dict): Available days for each doctor.
               - all_doctors (list): A list of all doctor names.
        Returns (None, None, None, None) on failure.
    """
    # 1. Initialize the four required data structures
    department_wise_doctors = {}
    doctor_consultation_fee = {}
    doctor_availability = {}
    all_doctors = []

    try:
        # Initialize Firebase Admin SDK
        # Ensure the path to your service account key is correct
        cred = credentials.Certificate("firebase_data_fetching/docbooking.json")
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        print("Successfully connected to Firestore.")
        print(f"Fetching details for clinic ID: '{clinic_id}'...")

        # Get the main clinic document
        clinic_ref = db.collection('clinics').document(clinic_id)
        clinic_doc = clinic_ref.get()

        if not clinic_doc.exists:
            print(f"Error: No clinic found with ID '{clinic_id}'.")
            return None, None, None, None

        clinic_name = clinic_doc.to_dict().get('name', 'the hospital')
        print(f"Found clinic: '{clinic_name}'. Fetching doctor details...")

        # Access the 'doctors' subcollection and stream the documents
        doctors_ref = clinic_doc.reference.collection('doctors')
        doctor_docs = doctors_ref.stream()

        # Iterate through each doctor document to populate the dictionaries
        for doc in doctor_docs:
            doctor_data = doc.to_dict()
            doctor_name = doctor_data.get('name')
            
            # Skip if the doctor has no name
            if not doctor_name:
                continue

            specialization = doctor_data.get('specialization', 'General')
            consultation_fees = doctor_data.get('consultationFees', 'N/A')
            consultation_times = doctor_data.get('consultationTimes')

            # 2. Populate the four data structures

            # a) Add doctor name to the `all_doctors` list
            all_doctors.append(doctor_name)

            # b) Add doctor's fee to the `doctor_consultation_fee` dictionary
            doctor_consultation_fee[doctor_name] = consultation_fees

            # c) Group doctors by department in `department_wise_doctors`
            if specialization not in department_wise_doctors:
                department_wise_doctors[specialization] = []
            department_wise_doctors[specialization].append(doctor_name)

            # d) Determine available days for `doctor_availability`
            available_days = []
            if consultation_times and isinstance(consultation_times, dict):
                # A day is considered available if its schedule is not null/None
                for day, schedule in consultation_times.items():
                    if schedule is not None:
                        available_days.append(day)
            doctor_availability[doctor_name] = available_days

        if not all_doctors:
            print("Clinic found, but no doctors were located in its subcollection.")

    except FileNotFoundError:
        print("Error: 'firebase_data_fetching/docbooking.json' not found.")
        print("Please ensure the service account key file is in the correct path.")
        return None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None, None

    print(f"\nSuccessfully processed details for {len(all_doctors)} doctors.")
    
    # 3. Return the four populated data structures
    return department_wise_doctors, doctor_consultation_fee, doctor_availability, all_doctors

# --- Main Execution Block (for testing purposes) ---
if __name__ == "__main__":
    # Set the target Clinic ID here
    TARGET_CLINIC_ID = "QV070" 
    
    # Call the function and unpack the returned tuple of data structures
    departments, fees, availability, doctors = fetch_clinic_data(TARGET_CLINIC_ID)
    # print('\n','\n',departments,'\n','\n',fees,'\n','\n', availability,'\n','\n',doctors,'\n','\n')
    # Print each data structure for verification
    if doctors:  # Check if any data was fetched successfully
        print("\n" + "="*50)
        print("--- 1. All Doctors (List) ---")
        print(json.dumps(doctors, indent=2))
        
        print("\n" + "="*50)
        print("--- 2. Department-wise Doctors (Dictionary) ---")
        print(json.dumps(departments, indent=2))

        print("\n" + "="*50)
        print("--- 3. Doctor Consultation Fees (Dictionary) ---")
        print(json.dumps(fees, indent=2))

        print("\n" + "="*50)
        print("--- 4. Doctor Availability by Day (Dictionary) ---")
        print(json.dumps(availability, indent=2))
        print("\n" + "="*50)
    else:
        print("\nNo data was fetched. Please check the Clinic ID and Firebase connection.")