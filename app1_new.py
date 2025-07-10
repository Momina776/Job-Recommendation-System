from flask import Flask, render_template, request, redirect, url_for, flash, session, g, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_paginate import Pagination
from datetime import datetime, date
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# MongoDB Connection
try:
    client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017/'))
    # Test the connection
    client.server_info()
    print("Successfully connected to MongoDB!")
    db = client['job_prediction_db']
except Exception as e:
    print(f"Error connecting to MongoDB: {str(e)}")

# Collections
users_collection = db.users
jobs_collection = db.jobs
applications_collection = db.applications
reviews_collection = db.reviews

# File Upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key_here')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.name = f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}"
        self.email = user_data.get('email', '')
        self.user_type = user_data.get('user_type', '')

@login_manager.user_loader
def load_user(user_id):
    user_data = users_collection.find_one({'_id': ObjectId(user_id)})
    if user_data:
        return User(user_data)
    return None

# ... rest of your routes ...

@app.route('/apply/<job_id>', methods=['GET', 'POST'])
@login_required
def apply_job(job_id):
    try:
        if current_user.user_type != 'job_seeker':
            flash('Only job seekers can apply for jobs', 'danger')
            return redirect(url_for('job_details', job_id=job_id))
        
        # Convert string ID to ObjectId
        job_object_id = ObjectId(job_id)
        
        # Check if the job exists
        job = jobs_collection.find_one({'_id': job_object_id})
        if not job:
            flash('Job not found', 'danger')
            return redirect(url_for('jobs'))
        
        # Check if already applied
        existing_application = applications_collection.find_one({
            'job_id': job_object_id,
            'job_seeker_id': ObjectId(current_user.id)
        })
        
        if existing_application:
            flash('You have already applied for this job', 'warning')
            return redirect(url_for('job_details', job_id=job_id))
        
        if request.method == 'POST':
            # Create new application
            application_data = {
                'job_id': job_object_id,
                'job_seeker_id': ObjectId(current_user.id),
                'cover_letter': request.form.get('cover_letter', ''),
                'date_applied': datetime.utcnow(),
                'status': 'pending'
            }
            
            applications_collection.insert_one(application_data)
            flash('Application submitted successfully!', 'success')
            return redirect(url_for('job_details', job_id=job_id))
            
        # For GET request, redirect to job details page with the application form
        return redirect(url_for('job_details', job_id=job_id))
        
    except Exception as e:
        print(f"Error in apply_job: {str(e)}")
        flash('Error submitting application. Please try again.', 'danger')
        return redirect(url_for('job_details', job_id=job_id))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
