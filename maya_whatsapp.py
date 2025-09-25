from flask import Flask, jsonify, request, Response
import requests
import google.generativeai as genai
import logging
import json

import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- !! PASTE YOUR KEYS AND TOKENS HERE !! ---
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
WHAT_TOKEN = "YOUR_WHATSAPP_TOKEN_HERE"
VERIFY_TOKEN = "YOUR_VERIFY_TOKEN_HERE"
PHONE_NUMBER_ID = "YOUR_PHONE_NUMBER_ID_HERE"
FIREBASE_CREDENTIALS_PATH = "serviceAccountKey.json"

# --- A simple in-memory dictionary to track user conversation states ---
# Key: user_phone_number, Value: 'awaiting_location'
user_states = {}

SYSTEM_INSTRUCTIONS = """You are 'DocBot', a friendly and efficient AI assistant for finding healthcare services in Kochi.
Your tone should be helpful, clear, and professional. You must answer questions based ONLY on the context provided to you.
When given a list of clinics for a specific location, present them clearly to the user.
If a user asks a general question, answer it concisely.
Always end your messages with "Stay healthy! 🩺".
"""

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    logging.info("Firebase configured and connected successfully.")
except Exception as e:
    logging.error(f"Error initializing Firebase: {e}")
    db = None

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    logging.info("Gemini AI model configured.")
except Exception as e:
    logging.error(f"Error configuring Gemini AI: {e}")
    model = None

# --- NEW FUNCTION: Gets a unique list of all locations ---
def get_all_unique_locations_from_firebase():
    """Fetches all visible clinics and returns a list of unique locations."""
    if not db: return None
    try:
        logging.info("Querying Firestore for unique clinic locations.")
        # Only get clinics that are marked as visible
        docs = db.collection('clinics').where('isVisible', '==', True).stream()
        
        # Use a set to automatically handle uniqueness
        locations = set()
        for doc in docs:
            doc_data = doc.to_dict()
            if 'location' in doc_data:
                locations.add(doc_data['location'])
        
        logging.info(f"Found unique locations: {list(locations)}")
        return sorted(list(locations)) # Return a sorted list
    except Exception as e:
        logging.error(f"Error fetching unique locations from Firestore: {e}")
        return None

# --- NEW FUNCTION: Gets clinics filtered by a specific location ---
def get_clinics_by_location_from_firebase(location):
    """Fetches all visible clinics for a given location."""
    if not db: return None
    try:
        logging.info(f"Querying Firestore for clinics in location: {location}")
        # The query needs to be case-sensitive, so we might need to handle this better in the future
        docs = db.collection('clinics').where('isVisible', '==', True).where('location', '==', location).stream()
        
        clinic_names = []
        for doc in docs:
            doc_data = doc.to_dict()
            if 'name' in doc_data:
                clinic_names.append(doc_data['name'])
        
        logging.info(f"Found clinics in {location}: {clinic_names}")
        return clinic_names
    except Exception as e:
        logging.error(f"Error fetching clinics by location from Firestore: {e}")
        return None

def ai_response(user_question, context_data=None):
    if not model: return "I'm sorry, my AI brain is taking a little break. Please try again later."
        
    logging.info(f"Generating AI response for: '{user_question}'")
    prompt = f"{SYSTEM_INSTRUCTIONS}\n\n"
    if context_data:
        prompt += f"--- CONTEXT DATA ---\nHere is the information you need:\n{json.dumps(context_data, indent=2)}\n--- END CONTEXT DATA ---\n\n"
    prompt += f"Based on the context, answer the user's question: \"{user_question}\""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error during Gemini API call: {e}")
        return "I encountered an issue while processing your request. Please try again."

@app.route('/', methods=["GET"])
def check_webhook():
    # This function remains the same
    mode = request.args.get('hub.mode'); verify_token = request.args.get('hub.verify_token'); challenge = request.args.get('hub.challenge')
    if mode and verify_token and mode == 'subscribe' and verify_token == VERIFY_TOKEN:
        return Response(challenge, 200)
    else: return Response("Webhook verification failed.", 403)

@app.route('/', methods=["POST"])
def send_message():
    body = request.get_json()
    logging.debug(f"Webhook payload: {json.dumps(body, indent=2)}")
    try:
        if (body.get("object") and body.get("entry") and body["entry"][0].get("changes") and
            body["entry"][0]["changes"][0].get("value") and body["entry"][0]["changes"][0]["value"].get("messages") and
            body["entry"][0]["changes"][0]["value"]["messages"][0]):
            
            message_info = body["entry"][0]["changes"][0]['value']["messages"][0]
            sender = message_info["from"]
            user_question = message_info["text"]["body"]
            logging.info(f"Message from {sender}: '{user_question}'")

            response_text = ""
            
            # --- START: CONVERSATIONAL LOGIC ---
            # Check if we are waiting for a location from this user
            if user_states.get(sender) == 'awaiting_location':
                chosen_location = user_question.strip()
                logging.info(f"User {sender} has chosen location: {chosen_location}")
                
                # Get clinics for the chosen location
                clinics_in_location = get_clinics_by_location_from_firebase(chosen_location)
                
                if clinics_in_location:
                    context_data = {"clinics_in_location": clinics_in_location}
                    # We create a new "question" for the AI based on the context
                    ai_question = f"List the clinics in {chosen_location}"
                    response_text = ai_response(ai_question, context_data)
                else:
                    response_text = f"I'm sorry, I couldn't find any clinics in '{chosen_location}'. Please check the spelling or choose from the list I provided. Stay healthy! 🩺"
                
                # IMPORTANT: Reset the user's state after fulfilling the request
                del user_states[sender]

            # --- If not in a conversation, check for new requests ---
            else:
                question_lower = user_question.lower()
                if "clinic" in question_lower or "clinics" in question_lower:
                    locations = get_all_unique_locations_from_firebase()
                    if locations:
                        # Format the locations list for the user
                        location_list_text = "\n".join([f"- {loc}" for loc in locations])
                        response_text = f"Of course! In which location would you like to find a clinic?\n\nPlease choose from the following:\n{location_list_text}"
                        
                        # IMPORTANT: Set the user's state to remember the context
                        user_states[sender] = 'awaiting_location'
                        logging.info(f"Set state for {sender} to 'awaiting_location'")
                    else:
                        response_text = "I couldn't find any clinic locations at the moment. Please try again later. Stay healthy! 🩺"
                else:
                    # If it's a general question, just get a direct AI response
                    response_text = ai_response(user_question)
            # --- END: CONVERSATIONAL LOGIC ---
            
            # Send the composed message back to the user
            url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
            headers = {"Authorization": f"Bearer {WHAT_TOKEN}", "Content-Type": "application/json"}
            data = {"messaging_product": "whatsapp", "to": sender, "type": "text", "text": {"body": response_text}}
            resp = requests.post(url, json=data, headers=headers)
            logging.info(f"Message sent. Status: {resp.status_code}, Body: {resp.text}")
            
    except Exception as e:
        logging.error(f"Error processing webhook payload: {e}")
    
    return Response(status=200)

if __name__ == '__main__':
    port = 5001
    app.run(host='0.0.0.0', port=port)