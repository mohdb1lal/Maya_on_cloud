import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter

# Global variable to store the Firestore client
db = None

def initialize_firebase():
    """
    Initialize Firebase Admin SDK (call this once at the start)
    """
    global db
    try:
        # Avoid re-initializing the app if already initialized
        if not firebase_admin._apps:
            cred = credentials.Certificate("firebase_data_fetching/docbooking.json")
            firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        print("Successfully connected to Firestore.")
        return True
    except FileNotFoundError:
        print("Error: 'docbooking.json' not found.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        return False


def convert_24hr_to_12hr(time_24):
    """
    Convert 24-hour time format to 12-hour format with AM/PM.
    
    Args:
        time_24 (float): Time in 24-hour format (e.g., 18.5 for 6:30 PM)
    
    Returns:
        str: Time in 12-hour format (e.g., "6:30 PM")
    """
    try:
        hours = int(time_24)
        minutes = int((time_24 - hours) * 60)
        
        # Determine AM/PM
        period = "AM" if hours < 12 else "PM"
        
        # Convert to 12-hour format
        display_hour = hours if hours <= 12 else hours - 12
        display_hour = 12 if display_hour == 0 else display_hour
        
        # Format minutes
        return f"{display_hour}:{minutes:02d} {period}"
    except:
        return str(time_24)


def format_availability_schedule(consultation_times):
    """
    Format the consultationTimes map into readable text format.
    
    Args:
        consultation_times (dict): Raw consultationTimes from Firestore
    
    Returns:
        list: List of readable availability strings
              ["Monday: 6:00 PM - 6:30 PM (1 slots)", ...]
    """
    formatted_schedule = []
    
    if not consultation_times:
        return ["No availability schedule found"]
    
    # Days order for sorting
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    for day in days_order:
        if day in consultation_times:
            day_schedule = consultation_times[day]
            
            # Check if day has None value (not available)
            if day_schedule is None:
                formatted_schedule.append(f"{day}: Not Available")
                continue
            
            # Process each session for the day
            sessions = []
            for session_key, session_data in day_schedule.items():
                if session_data and isinstance(session_data, dict):
                    from_time = session_data.get('from', 0)
                    to_time = session_data.get('to', 0)
                    token_limit = session_data.get('tokenLimit', 0)
                    
                    from_12hr = convert_24hr_to_12hr(from_time)
                    to_12hr = convert_24hr_to_12hr(to_time)
                    
                    sessions.append(f"{from_12hr} - {to_12hr} ({token_limit} slots)")
            
            if sessions:
                formatted_schedule.append(f"{day}: {', '.join(sessions)}")
            else:
                formatted_schedule.append(f"{day}: Not Available")
    
    return formatted_schedule


def get_doctors_list(clinic_id):
    """
    Get list of all doctors in a clinic with their names and specializations.
    
    Args:
        clinic_id (str): The clinic document ID (e.g., 'QV070')
    
    Returns:
        list: List of dictionaries containing doctor name and specialization
              [{"name": "Dr.Samuel Koshy", "specialization": "Pediatrics"}, ...]
    """
    doctors_list = []
    
    try:
        # Reference to the clinic document
        clinic_ref = db.collection('clinics').document(clinic_id)
        
        # Access the doctors subcollection
        doctors_ref = clinic_ref.collection('doctors')
        doctor_docs = doctors_ref.stream()
        
        # Iterate through all doctor documents
        for doc in doctor_docs:
            doctor_data = doc.to_dict()
            
            if 'name' in doctor_data and 'specialization' in doctor_data:
                doctors_list.append({
                    "name": doctor_data['name'],
                    "specialization": doctor_data['specialization']
                })
        
        if not doctors_list:
            print(f"No doctors found in clinic '{clinic_id}'")
        else:
            print(f"Found {len(doctors_list)} doctors in clinic '{clinic_id}'")
            
    except Exception as e:
        print(f"Error fetching doctors list: {e}")
    
    return doctors_list


def get_doctor_availability(clinic_id, doctor_name):
    """
    Get availability schedule (days and time slots) for a specific doctor.
    Returns formatted readable text instead of raw map.
    
    Args:
        clinic_id (str): The clinic document ID (e.g., 'QV070')
        doctor_name (str): The doctor's name (e.g., 'Dr.Samuel Koshy')
    
    Returns:
        list: List of formatted availability strings in 12-hour format
              ["Monday: 6:00 PM - 6:30 PM (1 slots)", 
               "Tuesday: 6:00 PM - 6:30 PM (1 slots)", ...]
    """
    availability = []
    
    try:
        # Reference to the clinic's doctors subcollection
        clinic_ref = db.collection('clinics').document(clinic_id)
        doctors_ref = clinic_ref.collection('doctors')
        
        # Query for the specific doctor by name (using new filter syntax)
        query = doctors_ref.where(filter=FieldFilter('name', '==', doctor_name)).limit(1)
        doctor_docs = list(query.stream())
        
        if not doctor_docs:
            print(f"Doctor '{doctor_name}' not found in clinic '{clinic_id}'")
            return ["Doctor not found"]
        
        doctor_data = doctor_docs[0].to_dict()
        
        # Extract consultationTimes and format them
        if 'consultationTimes' in doctor_data:
            raw_consultation_times = doctor_data['consultationTimes']
            availability = format_availability_schedule(raw_consultation_times)
            print(f"Found availability schedule for '{doctor_name}'")
        else:
            print(f"No consultationTimes found for '{doctor_name}'")
            availability = ["No availability schedule found"]
            
    except Exception as e:
        print(f"Error fetching doctor availability: {e}")
        availability = [f"Error: {str(e)}"]
    
    return availability


def get_doctor_consultation_fees(clinic_id, doctor_name):
    """
    Get consultation fees for a specific doctor.
    
    Args:
        clinic_id (str): The clinic document ID (e.g., 'QV070')
        doctor_name (str): The doctor's name (e.g., 'Dr.Samuel Koshy')
    
    Returns:
        str: Consultation fee (e.g., "250"), or empty string if not found
    """
    consultation_fees = ""
    
    try:
        # Reference to the clinic's doctors subcollection
        clinic_ref = db.collection('clinics').document(clinic_id)
        doctors_ref = clinic_ref.collection('doctors')
        
        # Query for the specific doctor by name (using new filter syntax)
        query = doctors_ref.where(filter=FieldFilter('name', '==', doctor_name)).limit(1)
        doctor_docs = list(query.stream())
        
        if not doctor_docs:
            print(f"Doctor '{doctor_name}' not found in clinic '{clinic_id}'")
            return ""
        
        doctor_data = doctor_docs[0].to_dict()
        
        # Extract consultationFees
        if 'consultationFees' in doctor_data:
            consultation_fees = doctor_data['consultationFees']
            print(f"Consultation fees for '{doctor_name}': ₹{consultation_fees}")
        else:
            print(f"No consultationFees found for '{doctor_name}'")
            
    except Exception as e:
        print(f"Error fetching consultation fees: {e}")
    
    return consultation_fees


def get_clinic_name(clinic_id):
    """
    Get the clinic name from the clinic ID.
    
    Args:
        clinic_id (str): The clinic document ID (e.g., 'QV070')
    
    Returns:
        str: Clinic name (e.g., "Jaza Healthcare"), or empty string if not found
    """
    try:
        clinic_ref = db.collection('clinics').document(clinic_id)
        clinic_doc = clinic_ref.get()
        
        if clinic_doc.exists:
            clinic_data = clinic_doc.to_dict()
            if 'name' in clinic_data:
                return clinic_data['name']
        else:
            print(f"Clinic with ID '{clinic_id}' not found")
            
    except Exception as e:
        print(f"Error fetching clinic name: {e}")
    
    return ""


def add_booking(clinic_id, booking_data):
    """
    Add a new booking to the clinic's bookings subcollection.
    
    Args:
        clinic_id (str): The clinic document ID (e.g., 'QV072')
        booking_data (dict): Dictionary containing all booking information
                            Required fields:
                            {
                                "patientName": "Test",
                                "phoneNumber": "+918078336549",
                                "age": 18,
                                "sex": "Male",
                                "doctorName": "Dr. Moideen Babu Perayil",
                                "doctorId": "0JRa28hpGhcsHZm7GPK8",
                                "specialization": "Neonatology",
                                "bookingDate": "2025-06-26",
                                "appointmentTime": "[9:00 AM - 4:30 PM]",
                                "selectedSession": "[Session 1]",
                                "token": 0,
                                "bookingStatus": "upcoming",
                                "paymentStatus": "unpaid",
                                "paymentMethod": "offline",
                                "paymentAmount": 2,
                                "totalPrice": 2,
                                "confirmedBooking": True,
                                "userEmail": "",
                                "uid": "V43GKjLbYYRo42Yiefu555SRquE3"
                                "bookingType": "Maya",
                            }
    
    Returns:
        str: The booking ID of the newly created booking, or empty string if failed
    """
    try:
        # Reference to the clinic's bookings subcollection
        clinic_ref = db.collection('clinics').document(clinic_id)
        bookings_ref = clinic_ref.collection('bookings')
        
        # Get clinic name for the booking
        clinic_doc = clinic_ref.get()
        clinic_name = ""
        if clinic_doc.exists:
            clinic_data = clinic_doc.to_dict()
            clinic_name = clinic_data.get('name', '')
        
        # Add clinic information to booking data
        booking_data['clinicId'] = clinic_id
        booking_data['clinicName'] = clinic_name
        
        # Add timestamp
        booking_data['timestamp'] = firestore.SERVER_TIMESTAMP
        
        # Create a new document with auto-generated ID
        new_booking_ref = bookings_ref.document()
        
        # Get the auto-generated booking ID
        booking_id = new_booking_ref.id
        
        # Add bookingId to the data
        booking_data['bookingId'] = booking_id
        
        # Save the booking
        new_booking_ref.set(booking_data)
        
        print(f"✅ Booking created successfully!")
        print(f"   Booking ID: {booking_id}")
        print(f"   Patient: {booking_data.get('patientName', '')}")
        print(f"   Doctor: {booking_data.get('doctorName', '')}")
        print(f"   Date: {booking_data.get('bookingDate', '')}")
        
        return booking_id
        
    except Exception as e:
        print(f"❌ Error adding booking: {e}")
        return ""


def add_booking_simple(clinic_id, patient_name, phone_number, doctor_name, doctor_id, 
                       booking_date, appointment_time, selected_session, 
                       consultation_fees, payment_method="online", age="", sex="", 
                       user_email="", uid="", token=0):
    """
    Simplified function to add a booking with essential parameters.
    
    Args:
        clinic_id (str): Clinic ID (e.g., 'QV072')
        patient_name (str): Patient's name
        phone_number (str): Patient's phone number (with country code like +918078336549)
        doctor_name (str): Doctor's name
        doctor_id (str): Doctor's ID
        booking_date (str): Booking date in format 'YYYY-MM-DD' (e.g., '2025-06-26')
        appointment_time (str): Appointment time range (e.g., '[9:00 AM - 4:30 PM]')
        selected_session (str): Session info (e.g., '[Session 1]')
        consultation_fees (int/str): Consultation fees amount
        payment_method (str, optional): Payment method ('online', 'cash', 'card'). Default: 'online'
        age (str/int, optional): Patient's age. Default: ""
        sex (str, optional): Patient's sex. Default: ""
        user_email (str, optional): User's email. Default: ""
        uid (str, optional): User ID. Default: ""
        token (int, optional): Token number. Default: 0
    
    Returns:
        str: The booking ID of the newly created booking
    """
    # Get doctor's specialization from the database
    specialization = ""
    try:
        clinic_ref = db.collection('clinics').document(clinic_id)
        doctors_ref = clinic_ref.collection('doctors')
        query = doctors_ref.where(filter=FieldFilter('name', '==', doctor_name)).limit(1)
        doctor_docs = list(query.stream())
        
        if doctor_docs:
            doctor_data = doctor_docs[0].to_dict()
            specialization = doctor_data.get('specialization', '')
    except Exception as e:
        print(f"Warning: Could not fetch doctor specialization: {e}")
    
    # Convert consultation fees to integer if it's a string
    if isinstance(consultation_fees, str):
        try:
            consultation_fees = int(consultation_fees)
        except:
            consultation_fees = 0
    
    # Prepare booking data
    booking_data = {
        "patientName": patient_name,
        "phoneNumber": phone_number,
        "age": age,
        "sex": sex,
        "doctorName": doctor_name,
        "doctorId": doctor_id,
        "specialization": specialization,
        "bookingDate": booking_date,
        "appointmentTime": appointment_time,
        "selectedSession": selected_session,
        "token": 0,
        "bookingStatus": "upcoming",
        "paymentStatus": "paid" if payment_method == "online" else "pending",
        "paymentMethod": payment_method,
        "paymentAmount": consultation_fees,
        "totalPrice": consultation_fees,
        "confirmedBooking": True if payment_method == "online" else False,
        "userEmail": user_email,
        "uid": uid,
        "bookingType" : 'Maya',
    }
    
    # Call the main add_booking function
    return add_booking(clinic_id, booking_data)


# def get_complete_clinic_data(clinic_id):
    """
    Get all clinic and doctor data in one function call.
    
    Args:
        clinic_id (str): The clinic document ID (e.g., 'QV070')
    
    Returns:
        dict: Complete clinic and doctor data
              {
                  "clinic_name": "Jaza Healthcare",
                  "clinic_id": "QV070",
                  "doctors": [
                      {
                          "name": "Dr.Samuel Koshy",
                          "specialization": "Pediatrics",
                          "consultation_fees": "250",
                          "availability": ["Monday: 6:00 PM - 6:30 PM (1 slots)", ...]
                      }
                  ]
              }
    """
    complete_data = {
        "clinic_name": "",
        "clinic_id": clinic_id,
        "doctors": []
    }
    
    # Get clinic name
    complete_data["clinic_name"] = get_clinic_name(clinic_id)
    
    # Get all doctors
    doctors_list = get_doctors_list(clinic_id)
    
    # For each doctor, get their complete data
    for doctor in doctors_list:
        doctor_name = doctor['name']
        
        doctor_info = {
            "name": doctor_name,
            "specialization": doctor['specialization'],
            "consultation_fees": get_doctor_consultation_fees(clinic_id, doctor_name),
            "availability": get_doctor_availability(clinic_id, doctor_name)
        }
        
        complete_data["doctors"].append(doctor_info)
    
    return complete_data


# --- Main execution ---
if __name__ == "__main__":
    # Initialize Firebase once
    if not initialize_firebase():
        print("Failed to initialize Firebase. Exiting.")
        exit()
    
    # Set your clinic ID
    CLINIC_ID = "QV070"
    
    print("\n" + "="*70)
    print("FETCHING CLINIC AND DOCTOR DATA")
    print("="*70)
    
    # Method 1: Using individual functions
    print("\n--- METHOD 1: Individual Functions ---")
    
    # 1. Get clinic name
    clinic_name = get_clinic_name(CLINIC_ID)
    print(f"\nClinic Name: {clinic_name}")
    print(f"Clinic ID: {CLINIC_ID}")
    
    # 2. Get list of all doctors
    print("\n--- Getting Doctors List ---")
    doctors_list = get_doctors_list(CLINIC_ID)
    print(f"\nDoctors List:")
    for doctor in doctors_list:
        print(f"  • {doctor['name']} ({doctor['specialization']})")
    
    # 3. Get availability for each doctor
    print("\n--- Getting Doctor Details ---")
    for doctor in doctors_list:
        doctor_name = doctor['name']
        print(f"\n{doctor_name} - {doctor['specialization']}")
        print("-" * 70)
        
        # Get consultation fees
        fees = get_doctor_consultation_fees(CLINIC_ID, doctor_name)
        print(f"  Consultation Fees: ₹{fees}")
        
        # Get availability
        availability = get_doctor_availability(CLINIC_ID, doctor_name)
        print(f"  Availability Schedule:")
        for schedule in availability:
            print(f"    • {schedule}")
    
    # Method 2: Add a new booking
    print("\n\n" + "="*70)
    print("--- ADDING NEW BOOKING ---")
    print("="*70)
    
    # # Example: Adding a booking using the simple method
    # new_booking_id = add_booking_simple(
    #     clinic_id=CLINIC_ID,
    #     patient_name="John Doe",
    #     phone_number="+919876543210",
    #     doctor_name="Dr. Moideen Babu Perayil",
    #     doctor_id="0JRa28hpGhcsHZm7GPK8",
    #     booking_date="2025-10-15",
    #     appointment_time="[9:00 AM - 4:30 PM]",
    #     selected_session="[Session 1]",
    #     consultation_fees=250,
    #     payment_method="online",
    #     age=30,
    #     sex="Male",
    #     token=1,
    # )
    
    # if new_booking_id:
    #     print(f"\n✅ Successfully created booking with ID: {new_booking_id}")
    
    print("\n" + "="*70)
    print("DATA FETCHING AND BOOKING COMPLETE")
    print("="*70)