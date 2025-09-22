import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceAccountKey.json")  # replace with your JSON key
firebase_admin.initialize_app(cred)

db = firestore.client()

def duplicate_collection(source_collection: str, target_collection: str):
    source_ref = db.collection(source_collection)
    docs = source_ref.stream()

    batch = db.batch()
    counter = 0
    batch_count = 1

    for doc in docs:
        new_doc_ref = db.collection(target_collection).document(doc.id)
        batch.set(new_doc_ref, doc.to_dict())
        counter += 1

        # Commit every 500 writes (Firestore batch limit)
        if counter % 500 == 0:
            batch.commit()
            print(f"✅ Batch {batch_count} committed (500 docs)")
            batch_count += 1
            batch = db.batch()

    # Commit remaining docs
    if counter % 500 != 0:
        batch.commit()
        print(f"✅ Final batch committed ({counter % 500} docs)")

    print(f"🎉 Duplicated {counter} documents from '{source_collection}' → '{target_collection}'")

# Example usage
duplicate_collection("clinics", "clinics_test")
