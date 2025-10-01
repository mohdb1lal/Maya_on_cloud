import firebase_admin
from firebase_admin import credentials, firestore

def fetch_doctor_availability(hospital_name):
    """
    Fetches doctor names and their availability from a Firestore database
    for a specific hospital from the 'clinics' collection.

    Args:
        hospital_name (str): The exact name of the hospital to query.

    Returns:
        dict: A dictionary with doctor names as keys and their availability
              (boolean) as values. Returns an empty dict if hospital or
              doctors are not found.
    """
    # Initialize the dictionary to store results
    doctor_availability_dict = {}

    try:
        # 1. Initialize Firebase Admin SDK
        # The script will look for 'serviceAccountKey.json' in the same folder.
        cred = credentials.Certificate("firebase-data-fetching/serviceAccountKey.json")
        
        # Avoid re-initializing the app if the script is run multiple times
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        # 2. Get the Firestore client
        db = firestore.client()
        
        print(f"Successfully connected to Firestore.")
        print(f"Searching for hospital: '{hospital_name}' in 'clinics' collection...")

        # 3. Find the hospital document by its 'name' field in the 'clinics' collection
        # UPDATED: Collection changed from 'hospitals' to 'clinics' based on your image.
        clinics_ref = db.collection('clinics')
        query = clinics_ref.where('name', '==', hospital_name).limit(1)
        hospital_docs = list(query.stream())

        if not hospital_docs:
            print(f"Error: No hospital found with the name '{hospital_name}' in the 'clinics' collection. Please check the name in your database.")
            return {}

        # 4. Access the 'doctors' subcollection of the found hospital
        hospital_doc = hospital_docs[0]
        print(f"Found hospital document. Fetching doctors from the 'doctors' subcollection...")
        
        # This part assumes the subcollection is named 'doctors' as shown in your previous setup.
        doctors_ref = hospital_doc.reference.collection('doctors')
        doctor_docs = doctors_ref.stream()

        # 5. Iterate through doctor documents and build the dictionary
        for doc in doctor_docs:
            doctor_data = doc.to_dict()
            # Ensure the document has the required fields
            if 'name' in doctor_data and 'availability' in doctor_data:
                doctor_name = doctor_data['name']
                is_available = doctor_data['availability']
                doctor_availability_dict[doctor_name] = is_available
        
        if not doctor_availability_dict:
            print("Hospital found, but no doctor documents were located in the 'doctors' subcollection.")

    except FileNotFoundError:
        print("Error: 'serviceAccountKey.json' not found.")
        print("Please download it from your Firebase project settings and place it in the same directory as this script.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}
        
    return doctor_availability_dict

# --- Main execution ---
if __name__ == "__main__":
    # UPDATED: Target hospital name now matches your database.
    # This must exactly match the 'name' field in your document.
    TARGET_HOSPITAL = "Zappa Hospital" 
    
    DOCTOR_AVAILABILITY = fetch_doctor_availability(TARGET_HOSPITAL)

    print("\n--- Final Result ---")
    # 6. Print the final dictionary
    print(DOCTOR_AVAILABILITY)