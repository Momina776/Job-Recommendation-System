from flask import Flask, render_template, request, redirect, url_for, flash, session, g, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_paginate import Pagination
from datetime import datetime, date
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from pymongo import MongoClient, ReturnDocument
from pymongo import errors as pymongo_errors
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Initialize app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key_here')
app.config['MONGO_URI'] = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')

# Initialize MongoDB connection
client = MongoClient(app.config['MONGO_URI'])
db = client['job_prediction_db']
jobs_collection = db.jobs
users_collection = db.users
applications_collection = db.applications
reviews_collection = db.reviews

@app.route('/job/<job_id>')
def job_details(job_id):
    try:
        print(f"Received job_id: {job_id}")
        
        # Try to convert string ID to ObjectId
        try:
            job_object_id = ObjectId(job_id)
            print(f"Successfully converted to ObjectId: {job_object_id}")
        except Exception as e:
            print(f"Error converting to ObjectId: {str(e)}")
            flash('Invalid job ID format.', 'danger')
            return redirect(url_for('jobs'))
        
        # Get job details with company information
        job = list(jobs_collection.aggregate([
            {'$match': {'_id': job_object_id}},
            {'$lookup': {
                'from': 'users',
                'localField': 'employer_id',
                'foreignField': '_id',
                'as': 'employer'
            }},
            {'$unwind': {'path': '$employer', 'preserveNullAndEmptyArrays': True}},
            {'$lookup': {
                'from': 'employer_profiles',
                'localField': 'employer_id',
                'foreignField': 'user_id',
                'as': 'employer_profile'
            }},
            {'$unwind': {'path': '$employer_profile', 'preserveNullAndEmptyArrays': True}},
            {'$project': {
                '_id': 1,
                'title': 1,
                'description': 1,
                'salary': 1,
                'location': 1,
                'status': 1,
                'date_posted': 1,
                'employer_id': 1,
                'company_name': '$employer_profile.company_name',
                'company_description': '$employer_profile.company_description',
                'industry': '$employer_profile.industry',
                'company_size': '$employer_profile.company_size',
                'formatted_date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$date_posted'}}
            }}
        ]))

        if not job:
            print(f"No job found with ID: {job_id}")
            flash('Job not found.', 'danger')
            return redirect(url_for('jobs'))

        job = job[0]
        print(f"Found job: {job['title']}")
        
        # Get application status if user is logged in and is a job seeker
        application_status = None
        application_date = None
        has_applied = False
        prediction_score = None
        
        if current_user.is_authenticated and current_user.user_type == 'job_seeker':
            # Get user's applications
            user_applications = list(applications_collection.find({
                'job_seeker_id': ObjectId(current_user.id)
            }))
            
            # Get user's profile
            user_profile = db.job_seeker_profiles.find_one({
                'user_id': ObjectId(current_user.id)
            })
            
            # Check current application status
            current_application = applications_collection.find_one({
                'job_id': job_object_id,
                'job_seeker_id': ObjectId(current_user.id)
            })
            
            if current_application:
                has_applied = True
                application_status = current_application.get('status')
                application_date = current_application.get('date_applied').strftime('%Y-%m-%d')

        # Get similar jobs
        similar_jobs = list(jobs_collection.aggregate([
            {'$match': {
                '_id': {'$ne': job_object_id},
                'status': 'open',
                '$or': [
                    {'title': {'$regex': job['title'], '$options': 'i'}},
                    {'description': {'$regex': job['title'], '$options': 'i'}}
                ]
            }},
            {'$limit': 3},
            {'$lookup': {
                'from': 'employer_profiles',
                'localField': 'employer_id',
                'foreignField': 'user_id',
                'as': 'employer_profile'
            }},
            {'$unwind': '$employer_profile'},
            {'$project': {
                '_id': 1,
                'title': 1,
                'company_name': '$employer_profile.company_name',
                'salary': 1
            }}
        ]))
        
        # Get reviews
        reviews = None
        if job.get('employer_id'):
            reviews = list(reviews_collection.find({
                'reviewee_id': job['employer_id']
            }).sort('date', -1).limit(5))
        
        return render_template('job_details.html',
                             job=job,
                             has_applied=has_applied,
                             application_status=application_status,
                             application_date=application_date,
                             similar_jobs=similar_jobs,
                             prediction_score=prediction_score,
                             reviews=reviews)
                             
    except Exception as e:
        import traceback
        print(f"Error in job_details: {str(e)}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        flash(f"Error loading job details: {str(e)}", 'danger')
        return redirect(url_for('jobs'))

if __name__ == '__main__':
    app.run(debug=True)
