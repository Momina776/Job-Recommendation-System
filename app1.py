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

# Allowed file extensions for resume uploads
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create app
app = Flask(__name__, 
           static_url_path='/static',
           static_folder='static',
           template_folder='templates')
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key_here')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['MONGO_URI'] = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Connect to MongoDB with retry logic
def connect_to_mongodb():
    retries = 3
    while retries > 0:
        try:
            client = MongoClient(app.config['MONGO_URI'], serverSelectionTimeoutMS=5000)
            # Test the connection
            client.server_info()
            db = client['job_prediction_db']
            print(f"Successfully connected to MongoDB at {app.config['MONGO_URI']}!")
            print(f"Available collections: {db.list_collection_names()}")
            return client, db
        except Exception as e:
            print(f"Error connecting to MongoDB (retries left: {retries-1}): {e}")
            print(f"MongoDB URI being used: {app.config['MONGO_URI']}")
            retries -= 1
            if retries > 0:
                time.sleep(2)  # Wait 2 seconds before retrying
    
    # If we get here, all retries failed
    print("Failed to connect to MongoDB after multiple attempts.")
    print("The application will continue with reduced functionality.")
    return None, None

# Initialize MongoDB connection
client, db = connect_to_mongodb()
if db is not None:
    # Initialize collections only if MongoDB is available
    users_collection = db.users
    jobs_collection = db.jobs
    applications_collection = db.applications
    reviews_collection = db.reviews
else:
    # Set collections to None to prevent errors
    users_collection = jobs_collection = applications_collection = reviews_collection = None

# Request logging
@app.before_request
def before_request():
    print(f"\nIncoming request: {request.method} {request.url}")
    print(f"Headers: {dict(request.headers)}")

# Error handling
@app.errorhandler(Exception)
def handle_error(error):
    print(f"Error: {str(error)}")
    return render_template('error.html', 
                         error_message="An unexpected error occurred. Please try again.",
                         retry_url=request.referrer)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html',
                         error_message="The requested page was not found.",
                         retry_url=url_for('index'))

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html',
                         error_message="An internal server error occurred. Please try again later.",
                         retry_url=request.referrer)

# --- User Class for Flask-Login ---
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

# --- Routes ---

@app.route('/')
def index():
    # Get featured jobs (most recent with complete information)
    featured_jobs = list(jobs_collection.aggregate([
        {'$match': {'status': 'open'}},
        {'$lookup': {
            'from': 'users',
            'localField': 'employer_id',
            'foreignField': '_id',
            'as': 'employer'
        }},
        {'$unwind': '$employer'},
        {'$lookup': {
            'from': 'employer_profiles',
            'localField': 'employer_id',
            'foreignField': 'user_id',
            'as': 'employer_profile'
        }},
        {'$unwind': '$employer_profile'},
        {'$sort': {'date_posted': -1}},
        {'$limit': 6},
        {'$project': {
            '_id': 1,
            'title': 1,
            'description': 1,
            'salary': 1,
            'location': 1,
            'date_posted': 1,
            'status': 1,
            'company_name': '$employer_profile.company_name',
            'formatted_date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$date_posted'}}
        }}
    ]))
    
    # Calculate platform statistics
    stats = {
        'total_jobs': jobs_collection.count_documents({'status': 'open'}),
        'total_companies': users_collection.count_documents({'user_type': 'employer'}),
        'total_applicants': users_collection.count_documents({'user_type': 'job_seeker'})
    }
    
    # Overall success rate
    pipeline = [
        {'$group': {
            '_id': None,
            'total_applications': {'$sum': 1},
            'accepted_applications': {'$sum': {'$cond': [{'$eq': ['$status', 'accepted']}, 1, 0]}}
        }}
    ]
    success_data = list(applications_collection.aggregate(pipeline))
    
    if success_data and success_data[0]['total_applications'] > 0:
        stats['success_rate'] = round((success_data[0]['accepted_applications'] / success_data[0]['total_applications']) * 100)
    else:
        stats['success_rate'] = 0
    
    return render_template('index.html', featured_jobs=featured_jobs, stats=stats)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user_data = users_collection.find_one({'email': email})
        
        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data)
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        user_type = request.form.get('user_type')

        # Validate required fields
        if not all([first_name, last_name, email, password, confirm_password, user_type]):
            flash('Please fill out all required fields.', 'danger')
            return redirect(url_for('register'))

        # Validate password confirmation
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        # Check if email exists
        if users_collection.find_one({'email': email}):
            flash('Email already exists. Please use a different email.', 'danger')
            return redirect(url_for('register'))

        try:
            # Insert user
            user_data = {
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'password': generate_password_hash(password),
                'user_type': user_type,
                'created_at': datetime.utcnow()
            }
            
            result = users_collection.insert_one(user_data)
            
            # Create user-type specific profile
            if user_type == 'employer':
                db.employer_profiles.insert_one({
                    'user_id': result.inserted_id,
                    'company_name': '',
                    'company_description': '',
                    'industry': '',
                    'company_size': ''
                })
            elif user_type == 'job_seeker':
                db.job_seeker_profiles.insert_one({
                    'user_id': result.inserted_id,
                    'skills': '',
                    'experience': '',
                    'resume_link': None
                })
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        
        except Exception as e:
            flash('An error occurred during registration.', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        if current_user.user_type == 'job_seeker':
            # Get recent applications with formatted dates
            applications = list(applications_collection.aggregate([
                {'$match': {'job_seeker_id': ObjectId(current_user.id)}},
                {'$sort': {'date_applied': -1}},
                {'$limit': 5},
                {'$lookup': {
                    'from': 'jobs',
                    'localField': 'job_id',
                    'foreignField': '_id',
                    'as': 'job'
                }},
                {'$unwind': '$job'},
                {'$lookup': {
                    'from': 'users',
                    'localField': 'job.employer_id',
                    'foreignField': '_id',
                    'as': 'employer'
                }},
                {'$unwind': '$employer'},
                {'$lookup': {
                    'from': 'employer_profiles',
                    'localField': 'job.employer_id',
                    'foreignField': 'user_id',
                    'as': 'employer_profile'
                }},
                {'$unwind': '$employer_profile'},
                {'$project': {
                    '_id': 1,
                    'status': 1,
                    'date_applied': 1,
                    'formatted_date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$date_applied'}},
                    'job_title': '$job.title',
                    'company_name': '$employer_profile.company_name'
                }}
            ]))
            
            # Get application statistics
            stats_pipeline = [
                {'$match': {'job_seeker_id': ObjectId(current_user.id)}},
                {'$group': {
                    '_id': None,
                    'total_applications': {'$sum': 1},
                    'accepted_applications': {'$sum': {'$cond': [{'$eq': ['$status', 'accepted']}, 1, 0]}}
                }}
            ]
            
            stats = list(applications_collection.aggregate(stats_pipeline))
            if stats:
                stats = stats[0]
            else:
                stats = {
                    'total_applications': 0,
                    'accepted_applications': 0,
                    'avg_rating': 0
                }

            # Get average rating from reviews
            rating_pipeline = [
                {'$match': {'reviewee_id': ObjectId(current_user.id)}},
                {'$group': {
                    '_id': None,
                    'avg_rating': {'$avg': '$rating'}
                }}
            ]
            ratings = list(reviews_collection.aggregate(rating_pipeline))
            if ratings:
                stats['avg_rating'] = ratings[0].get('avg_rating', 0)
            else:
                stats['avg_rating'] = 0
            
            return render_template('dashboard.html', applications=applications, stats=stats)
        
        else:
            # Employer dashboard
            posted_jobs = list(jobs_collection.aggregate([
                {'$match': {'employer_id': ObjectId(current_user.id)}},
                {'$lookup': {
                    'from': 'applications',
                    'localField': '_id',
                    'foreignField': 'job_id',
                    'as': 'applications'
                }},
                {'$project': {
                    '_id': 1,
                    'title': 1,
                    'status': 1,
                    'date_posted': 1,
                    'application_count': {'$size': '$applications'},
                    'formatted_date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$date_posted'}}
                }},
                {'$sort': {'date_posted': -1}}
            ]))
            
            # Employer stats pipeline
            stats_pipeline = [
                {'$match': {'employer_id': ObjectId(current_user.id)}},
                {'$group': {
                    '_id': None,
                    'active_jobs': {'$sum': 1},
                    'open_jobs': {'$sum': {'$cond': [{'$eq': ['$status', 'open']}, 1, 0]}}
                }}
            ]
            
            stats = list(jobs_collection.aggregate(stats_pipeline))
            
            # Get total applications
            apps_pipeline = [
                {'$match': {'employer_id': ObjectId(current_user.id)}},
                {'$group': {
                    '_id': None,
                    'total_applications': {'$sum': 1}
                }}
            ]
            
            apps_stats = list(applications_collection.aggregate(apps_pipeline))
            
            # Get company rating
            rating_pipeline = [
                {'$match': {'reviewee_id': ObjectId(current_user.id)}},
                {'$group': {
                    '_id': None,
                    'company_rating': {'$avg': '$rating'}
                }}
            ]
            
            ratings = list(reviews_collection.aggregate(rating_pipeline))
            
            # Combine all stats
            final_stats = {
                'active_jobs': stats[0]['active_jobs'] if stats else 0,
                'open_jobs': stats[0]['open_jobs'] if stats else 0,
                'total_applications': apps_stats[0]['total_applications'] if apps_stats else 0,
                'company_rating': ratings[0]['company_rating'] if ratings else 0
            }
            
            return render_template('dashboard.html', posted_jobs=posted_jobs, stats=final_stats)
    
    except Exception as e:
        print(f"Error in dashboard: {str(e)}")
        import traceback
        traceback.print_exc()
        flash('Error loading dashboard data. Please try again.', 'danger')
        return redirect(url_for('index'))

@app.route('/jobs')
def jobs():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 10
        skip = (page - 1) * per_page

        # Get filter parameters
        search = request.args.get('search', '')
        min_salary = request.args.get('min_salary', type=int)
        location = request.args.get('location', '')
        sort_by = request.args.get('sort_by', 'date')
        
        # Build query
        query = {'status': 'open'}
        
        if search:
            query['$or'] = [
                {'title': {'$regex': search, '$options': 'i'}},
                {'description': {'$regex': search, '$options': 'i'}}
            ]
        
        if min_salary:
            query['salary'] = {'$gte': min_salary}
        
        if location:
            query['location'] = {'$regex': location, '$options': 'i'}
        
        # Determine sort
        if sort_by == 'salary':
            sort = [('salary', -1)]
        else:  # Default to date
            sort = [('date_posted', -1)]

        # Get jobs with company information
        jobs = list(jobs_collection.aggregate([
            {'$match': query},
            {'$sort': dict(sort)},
            {'$skip': skip},
            {'$limit': per_page},
            {'$lookup': {
                'from': 'users',
                'localField': 'employer_id',
                'foreignField': '_id',
                'as': 'employer'
            }},
            {'$unwind': '$employer'},
            {'$lookup': {
                'from': 'employer_profiles',
                'localField': 'employer_id',
                'foreignField': 'user_id',
                'as': 'employer_profile'
            }},
            {'$unwind': '$employer_profile'},
            {'$project': {
                '_id': {'$toString': '$_id'},  # Convert ObjectId to string
                'title': 1,
                'description': 1,
                'salary': 1,
                'location': 1,
                'date_posted': 1,
                'employer_id': 1,
                'company_name': '$employer_profile.company_name',
                'formatted_date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$date_posted'}}
            }}
        ]))

        # Debug print
        print(f"Found {len(jobs)} jobs")
        for job in jobs:
            print(f"Job: {job['title']} - ID: {job['_id']}")

        # Count total jobs for pagination
        total_jobs = jobs_collection.count_documents(query)
        
        # Create pagination object
        pagination = Pagination(
            page=page,
            per_page=per_page,
            total=total_jobs,
            record_name='jobs',
            css_framework='bootstrap5'
        )

        return render_template('jobs.html', jobs=jobs, pagination=pagination)
    except Exception as e:
        print(f"Error in jobs route: {str(e)}")
        flash('Error loading jobs.', 'danger')
        return render_template('jobs.html', jobs=[], pagination=None)

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
              # ML prediction removed
            prediction_score = None

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
        
        # Get company reviews for sentiment analysis
        reviews = None
        if job.get('employer_id'):
            reviews = list(reviews_collection.find({
                'reviewee_id': job['employer_id']
            }).sort('date', -1).limit(5))
            
            if reviews:                # Analyze sentiment of reviews
                from ml_models import SentimentAnalysisModel
                sentiment_model = SentimentAnalysisModel()
                try:
                    for review in reviews:
                        sentiment_result = sentiment_model.predict_sentiment(review['review_text'])
                        review['sentiment'] = sentiment_result['sentiment']
                        review['confidence'] = sentiment_result['confidence']
                except Exception as e:
                    print(f"Error analyzing review sentiments: {str(e)}")
                    print(f"Found job: {job['title']}")
        
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

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    try:
        if request.method == 'POST':
            # Handle profile update
            user_id = ObjectId(current_user.id)
            first_name = request.form.get('first_name')
            last_name = request.form.get('last_name')
            email = request.form.get('email')
            
            # Update basic user information
            users_collection.update_one(
                {'_id': user_id},
                {'$set': {
                    'first_name': first_name,
                    'last_name': last_name,
                    'email': email
                }}
            )
            
            # Update user type specific information
            if current_user.user_type == 'job_seeker':
                skills = request.form.get('skills')
                experience = request.form.get('experience')
                resume = request.files.get('resume')
                
                # Handle resume upload if provided
                update_data = {
                    'skills': skills,
                    'experience': experience
                }
                
                if resume and resume.filename:
                    if not allowed_file(resume.filename):
                        flash('Invalid file type. Allowed types are: PDF, DOC, DOCX', 'danger')
                        return redirect(url_for('profile'))
                    
                    filename = secure_filename(f"{current_user.id}_{resume.filename}")
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    resume.save(filepath)
                    update_data['resume_link'] = filename
                
                db.job_seeker_profiles.update_one(
                    {'user_id': user_id},
                    {'$set': update_data},
                    upsert=True
                )
            
            elif current_user.user_type == 'employer':
                company_name = request.form.get('company_name')
                company_description = request.form.get('company_description')
                industry = request.form.get('industry')
                company_size = request.form.get('company_size')
                
                db.employer_profiles.update_one(
                    {'user_id': user_id},
                    {'$set': {
                        'company_name': company_name,
                        'company_description': company_description,
                        'industry': industry,
                        'company_size': company_size
                    }},
                    upsert=True
                )
            
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('profile'))
        
        # Get user data for display
        user_data = users_collection.find_one({'_id': ObjectId(current_user.id)})
        
        if current_user.user_type == 'job_seeker':
            profile_data = db.job_seeker_profiles.find_one({'user_id': ObjectId(current_user.id)})
            
            # Get application stats
            pipeline = [
                {'$match': {'job_seeker_id': ObjectId(current_user.id)}},
                {'$group': {
                    '_id': None,
                    'total': {'$sum': 1},
                    'accepted': {'$sum': {'$cond': [{'$eq': ['$status', 'accepted']}, 1, 0]}}
                }}
            ]
            stats = list(applications_collection.aggregate(pipeline))
            
            success_rate = 0
            if stats and stats[0]['total'] > 0:
                success_rate = (stats[0]['accepted'] / stats[0]['total']) * 100
        else:
            profile_data = db.employer_profiles.find_one({'user_id': ObjectId(current_user.id)})
            
            # Get hiring stats
            pipeline = [
                {'$match': {'employer_id': ObjectId(current_user.id)}},
                {'$lookup': {
                    'from': 'applications',
                    'localField': '_id',
                    'foreignField': 'job_id',
                    'as': 'applications'
                }},
                {'$unwind': '$applications'},
                {'$group': {
                    '_id': None,
                    'total': {'$sum': 1},
                    'hired': {'$sum': {'$cond': [{'$eq': ['$applications.status', 'accepted']}, 1, 0]}}
                }}
            ]
            stats = list(jobs_collection.aggregate(pipeline))
            
            hire_rate = 0
            if stats and stats[0]['total'] > 0:
                hire_rate = (stats[0]['hired'] / stats[0]['total']) * 100
        
        # Get reviews
        reviews = list(reviews_collection.aggregate([
            {'$match': {'reviewee_id': ObjectId(current_user.id)}},
            {'$lookup': {
                'from': 'users',
                'localField': 'reviewer_id',
                'foreignField': '_id',
                'as': 'reviewer'
            }},
            {'$unwind': '$reviewer'},
            {'$project': {
                '_id': 1,
                'rating': 1,
                'review_text': 1,
                'reviewer_name': {'$concat': ['$reviewer.first_name', ' ', '$reviewer.last_name']},
                'review_date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$review_date'}}
            }},
            {'$sort': {'review_date': -1}},
            {'$limit': 5}
        ]))
        
        # Calculate average rating
        rating_data = list(reviews_collection.aggregate([
            {'$match': {'reviewee_id': ObjectId(current_user.id)}},
            {'$group': {
                '_id': None,
                'avg_rating': {'$avg': '$rating'},
                'review_count': {'$sum': 1}
            }}
        ]))
        
        if rating_data:
            rating_data = rating_data[0]
            avg_rating = rating_data['avg_rating']
            review_count = rating_data['review_count']
        else:
            avg_rating = 0
            review_count = 0
        
        return render_template('profile.html',
                            user_data=user_data,
                            profile_data=profile_data,
                            reviews=reviews,
                            success_rate=success_rate if current_user.user_type == 'job_seeker' else None,
                            hire_rate=hire_rate if current_user.user_type == 'employer' else None,
                            avg_rating=avg_rating,
                            review_count=review_count)
    
    except Exception as e:
        flash('Error loading profile.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/post-job', methods=['GET', 'POST'])
@login_required
def post_job():
    if current_user.user_type != 'employer':
        flash('Only employers can post jobs', 'danger')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        try:
            # Get form data
            title = request.form.get('title')
            description = request.form.get('description')
            salary = float(request.form.get('salary'))
            location = request.form.get('location')
            
            # Insert job
            job_data = {
                'employer_id': ObjectId(current_user.id),
                'title': title,
                'description': description,
                'salary': salary,
                'location': location,
                'date_posted': datetime.utcnow(),
                'status': 'open'
            }
            
            result = jobs_collection.insert_one(job_data)
            flash('Job posted successfully!', 'success')
            return redirect(url_for('job_details', job_id=str(result.inserted_id)))
            
        except Exception as e:
            print(f"Error posting job: {str(e)}")
            flash('Error posting job.', 'danger')
    
    return render_template('post_job.html')

@app.route('/manage-applications/<job_id>')
@login_required
def manage_applications(job_id):
    if current_user.user_type != 'employer':
        flash('Only employers can manage applications', 'warning')
        return redirect(url_for('dashboard'))
    
    try:
        # Verify the job belongs to the current employer
        job = jobs_collection.find_one({
            '_id': ObjectId(job_id),
            'employer_id': ObjectId(current_user.id)
        })
        
        if not job:
            flash('Job not found or unauthorized.', 'danger')
            return redirect(url_for('dashboard'))
        
        # Get all applications for this job with applicant details and ML predictions
        applications = list(applications_collection.aggregate([
            {'$match': {'job_id': ObjectId(job_id)}},
            {'$lookup': {
                'from': 'users',
                'localField': 'job_seeker_id',
                'foreignField': '_id',
                'as': 'applicant'
            }},
            {'$unwind': '$applicant'},
            {'$lookup': {
                'from': 'job_seeker_profiles',
                'localField': 'job_seeker_id',
                'foreignField': 'user_id',
                'as': 'profile'
            }},
            {'$unwind': '$profile'},
            {'$project': {
                '_id': 1,
                'status': 1,
                'date_applied': 1,
                'cover_letter': 1,
                'applicant_name': {'$concat': ['$applicant.first_name', ' ', '$applicant.last_name']},
                'applicant_email': '$applicant.email',
                'skills': '$profile.skills',
                'experience': '$profile.experience',
                'resume_link': '$profile.resume_link',
                'formatted_date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$date_applied'}}
            }}
        ]))
        
        # Get ML predictions for each applicant
        from ml_models import JobPredictionModel
        job_model = JobPredictionModel()
        
        for application in applications:
            try:
                # Get applicant's history
                applicant_history = list(applications_collection.find({
                    'job_seeker_id': application['_id']
                }))
                
                # Get prediction
                features = job_model.prepare_features(applicant_history, {
                    'skills': application['skills'],
                    'experience': application['experience']
                })
                prediction = job_model.predict_success_probability(features)
                application['prediction_score'] = int(prediction * 100)
            except Exception as e:
                print(f"Error getting prediction for application {application['_id']}: {str(e)}")
                application['prediction_score'] = None
        
        return render_template('manage_applications.html', 
                             job=job, 
                             applications=applications)
    
    except Exception as e:
        print(f"Error managing applications: {str(e)}")
        flash('Error loading applications.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/my-applications')
@login_required
def my_applications():
    if current_user.user_type != 'job_seeker':
        flash('Only job seekers can view applications', 'warning')
        return redirect(url_for('dashboard'))
    
    try:
        # Get all applications with job and company details
        applications = list(applications_collection.aggregate([
            {'$match': {'job_seeker_id': ObjectId(current_user.id)}},
            {'$lookup': {
                'from': 'jobs',
                'localField': 'job_id',
                'foreignField': '_id',
                'as': 'job'
            }},
            {'$unwind': '$job'},
            {'$lookup': {
                'from': 'employer_profiles',
                'localField': 'job.employer_id',
                'foreignField': 'user_id',
                'as': 'employer'
            }},
            {'$unwind': '$employer'},
            {'$sort': {'date_applied': -1}},
            {'$project': {
                '_id': 1,
                'status': 1,
                'date_applied': 1,
                'cover_letter': 1,
                'job_title': '$job.title',
                'job_salary': '$job.salary',
                'company_name': '$employer.company_name',
                'formatted_date': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$date_applied'}}
            }}
        ]))
        
        return render_template('my_applications.html', applications=applications)
    
    except Exception as e:
        print(f"Error loading applications: {str(e)}")
        flash('Error loading applications. Please try again.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/apply/<job_id>', methods=['POST'])
@login_required
def apply_job(job_id):
    print("\n=== Job Application Submission ===")
    print(f"Job ID: {job_id}")
    print(f"User ID: {current_user.id}")
    print(f"User Type: {current_user.user_type}")
    
    # Log request headers for debugging AJAX/form submission
    print("\nRequest Headers:")
    for header, value in request.headers.items():
        print(f"{header}: {value}")
        
    # Log form data with safe length limits
    print("\nForm Data:")
    for key, value in request.form.items():
        if key == 'cover_letter':
            print(f"cover_letter length: {len(value)}")
            print(f"cover_letter preview: {value[:100]}...")
        else:
            print(f"{key}: {value}")
      # Validate request content type
    if not request.content_type:
        print("Error: No content type specified")
        return jsonify({
            'success': False,
            'message': 'Invalid request format',
            'error': 'INVALID_REQUEST'
        }), 400
        
    valid_content_types = [
        'application/x-www-form-urlencoded',
        'multipart/form-data'
    ]
    
    if not any(content_type in request.content_type for content_type in valid_content_types):
        print(f"Error: Invalid content type: {request.content_type}")
        return jsonify({
            'success': False,
            'message': f'Invalid request format. Must be one of: {", ".join(valid_content_types)}',
            'error': 'INVALID_REQUEST'
        }), 400
      # Check MongoDB connection
    if db is None:
        print("Error: MongoDB connection not available")
        return jsonify({
            'success': False,
            'message': 'Database connection error. Please try again later.',
            'error': 'CONNECTION_ERROR'
        }), 503
    
    # Validate user type
    if current_user.user_type != 'job_seeker':
        print("Error: User is not a job seeker")
        return jsonify({
            'success': False, 
            'message': 'Only job seekers can apply for jobs',
            'error': 'UNAUTHORIZED',
            'redirectUrl': url_for('dashboard')
        }), 403
    
    try:
        # Validate job_id format
        try:
            job_object_id = ObjectId(job_id)
            print(f"Valid ObjectId: {job_object_id}")
        except Exception as e:
            print(f"Invalid job_id format: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Invalid job ID format.',
                'error': 'INVALID_ID',
                'redirectUrl': url_for('jobs')
            }), 400

        # Check if job exists and is open
        print("\nChecking job status...")
        try:
            job = jobs_collection.find_one({
                '_id': job_object_id,
                'status': 'open'
            })
            if job:
                print(f"Found job: {job.get('title', 'Unknown Title')}")
            else:
                print("Job not found or not open")
                return jsonify({
                    'success': False,
                    'message': 'Job not found or no longer accepting applications.',
                    'error': 'JOB_NOT_FOUND',
                    'redirectUrl': url_for('jobs')
                }), 404
        except pymongo_errors.ConnectionFailure as e:
            print(f"MongoDB connection error while checking job: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Could not connect to the database. Please try again later.',
                'error': 'CONNECTION_ERROR'
            }), 503

        # Validate form data
        required_fields = ['cover_letter']
        missing_fields = [field for field in required_fields if not request.form.get(field)]
        if missing_fields:
            print(f"Missing required fields: {missing_fields}")
            return jsonify({
                'success': False,
                'message': f"Missing required fields: {', '.join(missing_fields)}",
                'error': 'VALIDATION_ERROR'
            }), 400

        # Get and validate cover letter
        print("\nValidating cover letter...")
        cover_letter = request.form.get('cover_letter', '').strip()
        print(f"Cover letter validation: {len(cover_letter)} chars")
        
        # Validate cover letter content
        if not cover_letter:
            print("Error: Empty cover letter")
            return jsonify({
                'success': False,
                'message': 'Cover letter cannot be empty.',
                'error': 'VALIDATION_ERROR'
            }), 400
            
        if len(cover_letter) < 100:
            print("Error: Cover letter too short")
            return jsonify({
                'success': False,
                'message': 'Your cover letter must be at least 100 characters long.',
                'error': 'VALIDATION_ERROR'
            }), 400
            
        if len(cover_letter) > 5000:
            print("Error: Cover letter too long")
            return jsonify({
                'success': False,
                'message': 'Your cover letter cannot exceed 5000 characters.',
                'error': 'VALIDATION_ERROR'
            }), 400

        # Check for duplicate application
        print("\nChecking for existing application...")
        try:
            existing_application = applications_collection.find_one({
                'job_id': job_object_id,
                'job_seeker_id': ObjectId(current_user.id)
            })
            if existing_application:
                print("Found existing application")
                return jsonify({
                    'success': False,
                    'message': 'You have already applied for this job.',
                    'error': 'DUPLICATE_APPLICATION',
                    'redirectUrl': url_for('job_details', job_id=job_id)
                }), 409
        except pymongo_errors.ConnectionFailure as e:
            print(f"MongoDB connection error while checking existing application: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Could not connect to the database. Please try again later.',
                'error': 'CONNECTION_ERROR'
            }), 503
        
        # Create application document
        print("\nPreparing application document...")
        application_data = {
            'job_id': job_object_id,
            'job_seeker_id': ObjectId(current_user.id),
            'employer_id': job['employer_id'],
            'date_applied': datetime.utcnow(),
            'status': 'pending',
            'cover_letter': cover_letter
        }
        print("Application data prepared")
        
        # Insert application with retry logic
        print("\nAttempting to insert application...")
        max_retries = 3
        retry_delay = 1  # seconds
        last_error = None
        
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1} of {max_retries}")
                result = applications_collection.insert_one(application_data)
                
                if result.inserted_id:
                    print(f"Success! Application inserted with ID: {result.inserted_id}")
                    return jsonify({
                        'success': True,
                        'message': 'Application submitted successfully!',
                        'redirectUrl': url_for('my_applications')
                    }), 201
                else:
                    print("Error: No inserted_id returned")
                    raise Exception('Failed to insert application: No inserted_id returned')
                    
            except pymongo_errors.DuplicateKeyError as e:
                print(f"Duplicate key error: {str(e)}")
                return jsonify({
                    'success': False,
                    'message': 'You have already applied for this job.',
                    'error': 'DUPLICATE_APPLICATION',
                    'redirectUrl': url_for('job_details', job_id=job_id)
                }), 409
                
            except pymongo_errors.ConnectionFailure as e:
                print(f"Connection error on attempt {attempt + 1}: {str(e)}")
                last_error = e
                if attempt == max_retries - 1:
                    raise
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                
            except pymongo_errors.WriteError as e:
                print(f"MongoDB write error: {str(e)}")
                if "duplicate key error" in str(e).lower():
                    return jsonify({
                        'success': False,
                        'message': 'You have already applied for this job.',
                        'error': 'DUPLICATE_APPLICATION'
                    }), 409
                return jsonify({
                    'success': False,
                    'message': 'Error saving application data.',
                    'error': 'DATABASE_ERROR'
                }), 500
                
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                last_error = e
                if attempt == max_retries - 1:
                    raise
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        # If we get here, all retries failed
        print("\nAll insertion attempts failed")
        raise last_error or Exception('Failed to insert application after multiple attempts')
        
    except pymongo_errors.ConnectionFailure as e:
        print(f"\nFatal connection error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Could not connect to the database. Please try again later.',
            'error': 'CONNECTION_ERROR'
        }), 503
        
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'An error occurred while submitting your application. Please try again.',
            'error': 'SERVER_ERROR'
        }), 500
@app.route('/update-application/<application_id>/<status>')
@login_required
def update_application_status(application_id, status):
    print(f"\n=== Updating Application Status ===")
    print(f"Application ID: {application_id}")
    print(f"New Status: {status}")
    print(f"User: {current_user.id} (Type: {current_user.user_type})")
    
    # Validate user type
    if current_user.user_type != 'employer':
        print("Error: User is not an employer")
        return jsonify({
            'success': False,
            'message': 'Only employers can update application status',
            'error': 'UNAUTHORIZED'
        }), 403
    
    # Validate status value
    valid_statuses = ['pending', 'accepted', 'rejected']
    if status not in valid_statuses:
        print(f"Error: Invalid status {status}")
        return jsonify({
            'success': False,
            'message': f'Invalid status. Must be one of: {", ".join(valid_statuses)}',
            'error': 'INVALID_STATUS'
        }), 400
    
    try:
        # Convert application_id to ObjectId
        try:
            application_object_id = ObjectId(application_id)
        except Exception as e:
            print(f"Invalid application_id format: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Invalid application ID format',
                'error': 'INVALID_ID'
            }), 400
        
        # Find the application and verify employer ownership
        application = applications_collection.find_one({
            '_id': application_object_id
        })
        
        if not application:
            print("Application not found")
            return jsonify({
                'success': False,
                'message': 'Application not found',
                'error': 'NOT_FOUND'
            }), 404
            
        # Verify the application is for a job owned by this employer
        job = jobs_collection.find_one({
            '_id': application['job_id'],
            'employer_id': ObjectId(current_user.id)
        })
        
        if not job:
            print("Unauthorized: Job not owned by current employer")
            return jsonify({
                'success': False,
                'message': 'You are not authorized to update this application',
                'error': 'UNAUTHORIZED'
            }), 403
            
        # Update application status
        result = applications_collection.update_one(
            {'_id': application_object_id},
            {'$set': {
                'status': status,
                'updated_at': datetime.utcnow()
            }}
        )
        
        if result.modified_count == 1:
            print(f"Successfully updated application {application_id} to {status}")
            
            # Get applicant details for notification
            applicant = users_collection.find_one({'_id': application['job_seeker_id']})
            if applicant:
                # TODO: Send email notification to applicant
                print(f"Would send email to {applicant.get('email')} about status update")
            
            return jsonify({
                'success': True,
                'message': f'Application status updated to {status}',
                'redirectUrl': url_for('manage_applications', job_id=str(job['_id']))
            }), 200
        else:
            print("No changes made to application")
            return jsonify({
                'success': False,
                'message': 'No changes made to application status',
                'error': 'NO_CHANGES'
            }), 400
            
    except pymongo_errors.ConnectionFailure as e:
        print(f"Database connection error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Database connection error. Please try again later.',
            'error': 'CONNECTION_ERROR'
        }), 503
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'An error occurred while updating the application status',
            'error': 'SERVER_ERROR'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000, host='127.0.0.1')