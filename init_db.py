from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from datetime import datetime
from bson import ObjectId

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['job_prediction_db']

# Clear existing collections
db.users.delete_many({})
db.jobs.delete_many({})
db.employer_profiles.delete_many({})
db.job_seeker_profiles.delete_many({})

# Create a test employer
employer_id = ObjectId()
employer_data = {
    '_id': employer_id,
    'first_name': 'Test',
    'last_name': 'Employer',
    'email': 'employer@test.com',
    'password': generate_password_hash('password123'),
    'user_type': 'employer',
    'created_at': datetime.utcnow()
}
db.users.insert_one(employer_data)

# Create employer profile
employer_profile = {
    'user_id': employer_id,
    'company_name': 'Test Company',
    'company_description': 'A test company',
    'industry': 'Technology',
    'company_size': '10-50'
}
db.employer_profiles.insert_one(employer_profile)

# Create some test jobs
jobs = [
    {
        '_id': ObjectId(),
        'employer_id': employer_id,
        'title': 'Software Engineer',
        'description': 'Looking for a skilled software engineer',
        'salary': 80000,
        'location': 'New York',
        'date_posted': datetime.utcnow(),
        'status': 'open'
    },
    {
        '_id': ObjectId(),
        'employer_id': employer_id,
        'title': 'Data Scientist',
        'description': 'Data scientist position available',
        'salary': 90000,
        'location': 'San Francisco',
        'date_posted': datetime.utcnow(),
        'status': 'open'
    }
]

db.jobs.insert_many(jobs)

print("Database initialized with test data!")
