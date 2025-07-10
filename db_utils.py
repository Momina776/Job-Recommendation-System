import os
from pymongo import MongoClient, errors
from datetime import datetime
import time
from dotenv import load_dotenv

def get_db_connection(max_retries=3, retry_delay=1):
    """Get MongoDB connection with retry logic"""
    load_dotenv()
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    
    for attempt in range(max_retries):
        try:
            client = MongoClient(mongo_uri, 
                               serverSelectionTimeoutMS=5000,
                               connectTimeoutMS=5000,
                               socketTimeoutMS=5000)
            # Verify connection
            client.server_info()
            return client, None
        except errors.ServerSelectionTimeoutError as e:
            if attempt == max_retries - 1:
                return None, f"Could not connect to MongoDB after {max_retries} attempts: {str(e)}"
            print(f"Connection attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        except Exception as e:
            return None, f"Error connecting to MongoDB: {str(e)}"

def initialize_collections(db):
    """Initialize MongoDB collections if they don't exist"""
    collections = ['users', 'jobs', 'applications', 'reviews', 'employer_profiles', 'job_seeker_profiles']
    existing_collections = db.list_collection_names()
    
    for collection in collections:
        if collection not in existing_collections:
            db.create_collection(collection)
            print(f"Created collection: {collection}")
            
            # Add indexes for common queries
            if collection == 'users':
                db[collection].create_index('email', unique=True)
            elif collection == 'jobs':
                db[collection].create_index([('title', 'text'), ('description', 'text')])
                db[collection].create_index('employer_id')
                db[collection].create_index('status')
            elif collection == 'applications':
                db[collection].create_index([('job_id', 1), ('job_seeker_id', 1)], unique=True)
                db[collection].create_index('status')
            elif collection == 'reviews':
                db[collection].create_index('reviewee_id')

def check_and_setup_database():
    """Initialize database and verify connection"""
    client, error = get_db_connection()
    if error:
        return False, error
    
    try:
        db = client['job_prediction_db']
        collections = ['users', 'jobs', 'applications', 'reviews', 'employer_profiles', 'job_seeker_profiles']
        
        # Create collections if they don't exist
        for collection in collections:
            if collection not in db.list_collection_names():
                db.create_collection(collection)
        
        # Create indexes
        db.users.create_index('email', unique=True)
        db.jobs.create_index([('title', 'text'), ('description', 'text')])
        db.applications.create_index([('job_id', 1), ('job_seeker_id', 1)], unique=True)
        
        return True, "Database setup completed successfully"
    except Exception as e:
        return False, f"Error setting up database: {str(e)}"
    finally:
        client.close()

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_database_connection():
    """
    Create a connection to MongoDB and return the database instance
    """
    try:
        # Get MongoDB URI from environment variable or use default
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
        
        # Create client with timeout
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        
        # Force a connection to verify it works
        client.server_info()
        
        # Get database
        db = client['job_prediction_db']
        return True, db, "Successfully connected to database"
    
    except ConnectionFailure as e:
        return False, None, f"Failed to connect to MongoDB. Error: {str(e)}"
    except Exception as e:
        return False, None, f"An error occurred while connecting to database: {str(e)}"

def check_and_setup_database():
    """
    Check database connection and setup initial collections if needed
    """
    success, db, message = get_database_connection()
    
    if not success:
        return False, message
    
    try:
        # Create collections if they don't exist
        collections = ['users', 'jobs', 'applications', 'reviews', 'employer_profiles', 'job_seeker_profiles']
        for collection in collections:
            if collection not in db.list_collection_names():
                db.create_collection(collection)
        
        return True, "Database setup completed successfully"
    except Exception as e:
        return False, f"Error setting up database: {str(e)}"
