import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceAccountKey.json")  # replace with your service account key
firebase_admin.initialize_app(cred)

db = firestore.client()

def get_collection_schema(collection_name: str, sample_limit: int = 100):
    """Scans documents to infer schema of a collection"""
    collection_ref = db.collection(collection_name)
    docs = collection_ref.limit(sample_limit).stream()  # limit for performance
    
    schema = {}

    for doc in docs:
        data = doc.to_dict()
        for field, value in data.items():
            dtype = type(value).__name__
            if field not in schema:
                schema[field] = set()
            schema[field].add(dtype)

    # Format schema
    print(f"📂 Schema for collection '{collection_name}' (sampled {sample_limit} docs):\n")
    for field, dtypes in schema.items():
        print(f"- {field}: {', '.join(sorted(dtypes))}")

# Example usage
get_collection_schema("clinics", sample_limit=200)