from db_utils import get_db_connection
import json
from datetime import datetime, UTC
import traceback

def test_connection_and_insertion():
    print("Starting MongoDB connection test...")
    # Test connection
    client, error = get_db_connection()
    if error:
        print(f"Connection error: {error}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False
    
    try:
        db = client['job_portal']  # Use your actual database name
        
        # Create a test document
        test_application = {
            "job_id": "test_job_id",
            "user_id": "test_user_id",            "cover_letter": "This is a test cover letter",
            "status": "pending",
            "created_at": datetime.now(UTC),
            "test_field": True
        }
        
        # Try to insert the document
        result = db.applications.insert_one(test_application)
        print(f"Test document inserted with ID: {result.inserted_id}")
        
        # Verify the document was inserted
        retrieved = db.applications.find_one({"_id": result.inserted_id})
        if retrieved:
            print("Successfully retrieved test document")
            
            # Clean up - delete the test document
            db.applications.delete_one({"_id": result.inserted_id})
            print("Test document cleaned up")
            return True
    except Exception as e:
        print(f"Error during test: {str(e)}")
        return False
    finally:
        client.close()

if __name__ == "__main__":
    success = test_connection_and_insertion()
    print(f"\nTest {'passed' if success else 'failed'}")
