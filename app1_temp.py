from flask import Flask, render_template, request, redirect, url_for, flash, session, g, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import sqlite3
from flask_paginate import Pagination
from datetime import datetime, date
import json
import os
from werkzeug.utils import secure_filename

# File Upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
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

DATABASE = 'FinalProject.db'  # Your database name

# --- Database Connection Setup ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# --- User Class for Flask-Login ---
class User(UserMixin):
    def __init__(self, user_id, name, email, user_type):
        self.id = user_id
        self.name = name
        self.email = email
        self.user_type = user_type

@login_manager.user_loader
def load_user(user_id):
    db = get_db()
    cur = db.execute('SELECT * FROM Users WHERE user_id = ?', (user_id,))
    user_row = cur.fetchone()
    if user_row:
        name = f"{user_row['first_name']} {user_row['last_name']}"
        return User(user_row['user_id'], name, user_row['email'], user_row['user_type'])
    return None

# --- Routes ---

@app.route('/')
def index():
    db = get_db()
    
    # Get featured jobs (most recent with complete information)
    featured_jobs = db.execute('''
        SELECT Jobs.*, Employer.company_name,
               strftime('%Y-%m-%d', Jobs.date_posted) as formatted_date
        FROM Jobs 
        JOIN Users ON Jobs.employer_id = Users.user_id
        JOIN Employer ON Users.user_id = Employer.user_id
        WHERE Jobs.status = 'open'
        ORDER BY Jobs.date_posted DESC
        LIMIT 6
    ''').fetchall()
    
    # Calculate platform statistics
    stats = {}
    
    # Total active jobs
    stats['total_jobs'] = db.execute('''
        SELECT COUNT(*) as count
        FROM Jobs
        WHERE status = 'open'
    ''').fetchone()['count']
    
    # Total companies (employers)
    stats['total_companies'] = db.execute('''
        SELECT COUNT(*) as count
        FROM Users
        WHERE user_type = 'employer'
    ''').fetchone()['count']
    
    # Total job seekers
    stats['total_applicants'] = db.execute('''
        SELECT COUNT(*) as count
        FROM Users
        WHERE user_type = 'job_seeker'
    ''').fetchone()['count']
    
    # Overall success rate
    success_data = db.execute('''
        SELECT 
            COUNT(*) as total_applications,
            SUM(CASE WHEN status = 'accepted' THEN 1 ELSE 0 END) as accepted_applications
        FROM Applications
    ''').fetchone()
    
    total_apps = success_data['total_applications']
    if total_apps > 0:
        stats['success_rate'] = round((success_data['accepted_applications'] / total_apps) * 100)
    else:
        stats['success_rate'] = 0
    
    return render_template('index.html', featured_jobs=featured_jobs, stats=stats)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        db = get_db()
        cur = db.execute('SELECT * FROM Users WHERE email = ? AND password = ?', (email, password))
        user_row = cur.fetchone()
        if user_row:
            name = f"{user_row['first_name']} {user_row['last_name']}"
            user = User(user_row['user_id'], name, user_row['email'], user_row['user_type'])
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

        # Insert user into the database
        db = get_db()
        try:
            db.execute(
                'INSERT INTO Users (first_name, last_name, email, password, user_type) '
                'VALUES (?, ?, ?, ?, ?)',
                (first_name, last_name, email, password, user_type)
            )
            db.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists. Please use a different email.', 'danger')
            return redirect(url_for('register'))
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
    db = get_db()
    
    if current_user.user_type == 'job_seeker':
        # Get recent applications with formatted dates
        applications = db.execute('''
            SELECT a.*, j.title, e.company_name,
                   strftime('%Y-%m-%d', a.date_applied) as formatted_date
            FROM Applications a
            JOIN Jobs j ON a.job_id = j.job_id
            JOIN Users u ON j.employer_id = u.user_id
            JOIN Employer e ON u.user_id = e.user_id
            WHERE a.job_seeker_id = ?
            ORDER BY a.date_applied DESC
            LIMIT 5
        ''', (current_user.id,)).fetchall()
        
        # Get application statistics
        stats = db.execute('''
            SELECT 
                COUNT(*) as total_applications,
                SUM(CASE WHEN status = 'accepted' THEN 1 ELSE 0 END) as accepted_applications,
                AVG(CASE WHEN r.rating IS NOT NULL THEN r.rating END) as avg_rating
            FROM Applications a
            LEFT JOIN Reviews r ON r.reviewee_id = ?
            WHERE a.job_seeker_id = ?
        ''', (current_user.id, current_user.id)).fetchone()
        
        return render_template('dashboard.html', 
                             applications=applications,
                             stats=stats)
    else:
        # Employer dashboard
        posted_jobs = db.execute('''
            SELECT j.*, 
                   strftime('%Y-%m-%d', j.date_posted) as formatted_date,
                   COUNT(a.application_id) as application_count
            FROM Jobs j
            LEFT JOIN Applications a ON j.job_id = a.job_id
            WHERE j.employer_id = ?
            GROUP BY j.job_id
            ORDER BY j.date_posted DESC
        ''', (current_user.id,)).fetchall()
        
        return render_template('dashboard.html', 
                             posted_jobs=posted_jobs)

@app.route('/jobs')
def jobs():
    db = get_db()
    page = request.args.get('page', 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    # Get filter parameters
    search = request.args.get('search', '')
    min_salary = request.args.get('min_salary', type=int)
    location = request.args.get('location', '')
    sort_by = request.args.get('sort_by', '')
    
    # Build base query with proper table aliases and joins
    base_query = '''
        SELECT j.*, e.company_name,
               strftime('%Y-%m-%d', j.date_posted) as formatted_date
        FROM Jobs AS j
        INNER JOIN Users AS u ON j.employer_id = u.user_id
        INNER JOIN Employer AS e ON u.user_id = e.user_id
        WHERE 1=1
    '''
    params = []

    # Add search filter
    if search:
        base_query += ''' AND (j.title LIKE ? OR e.company_name LIKE ?)'''
        params.extend(['%' + search + '%', '%' + search + '%'])

    # Add salary filter
    if min_salary:
        base_query += ' AND j.salary >= ?'
        params.append(min_salary)

    # Add location filter
    if location:
        base_query += ' AND j.location LIKE ?'
        params.append('%' + location + '%')

    # Add sorting
    if sort_by == 'date':
        base_query += ' ORDER BY j.date_posted DESC'
    elif sort_by == 'salary':
        base_query += ' ORDER BY j.salary DESC'
    else:
        base_query += ' ORDER BY j.date_posted DESC'

    # Add pagination
    query = base_query + ' LIMIT ? OFFSET ?'
    params.extend([per_page, offset])

    # Execute query
    jobs = db.execute(query, params).fetchall()

    # Count total jobs for pagination (reuse base query)
    count_query = '''
        SELECT COUNT(*) as count 
        FROM Jobs AS j
        INNER JOIN Users AS u ON j.employer_id = u.user_id
        INNER JOIN Employer AS e ON u.user_id = e.user_id
        WHERE j.status = 'open'
    '''
    
    # Add the same filters to count query
    count_params = []
    if search:
        count_query += ''' AND (j.title LIKE ? OR e.company_name LIKE ?)'''
        count_params.extend(['%' + search + '%', '%' + search + '%'])
    if min_salary:
        count_query += ' AND j.salary >= ?'
        count_params.append(min_salary)
    if location:
        count_query += ' AND j.location LIKE ?'
        count_params.append('%' + location + '%')

    total_jobs = db.execute(count_query, count_params).fetchone()['count']

    # Create pagination object
    pagination = Pagination(
        page=page,
        per_page=per_page,
        total=total_jobs,
        record_name='jobs',
        css_framework='bootstrap5'
    )

    return render_template('jobs.html', jobs=jobs, pagination=pagination)

@app.route('/job/<int:job_id>')
def job_details(job_id):
    db = get_db()
    # Get job details with company information using proper table aliases
    job = db.execute('''
        SELECT j.*, e.*, u.first_name, u.last_name, u.email,
               strftime('%Y-%m-%d', j.date_posted) as formatted_date
        FROM Jobs AS j
        INNER JOIN Users AS u ON j.employer_id = u.user_id
        INNER JOIN Employer AS e ON u.user_id = e.user_id
        WHERE j.job_id = ?
    ''', (job_id,)).fetchone()
    
    if not job:
        flash('Job not found.', 'danger')
        return redirect(url_for('jobs'))

    # Convert the job record to a mutable dictionary
    job = dict(job)
        
    # Get application status if user is logged in and is a job seeker
    application_status = None
    application_date = None
    if current_user.is_authenticated and current_user.user_type == 'job_seeker':
        application = db.execute('''
            SELECT status, 
                   strftime('%Y-%m-%d', date_applied) as formatted_date
            FROM Applications 
            WHERE job_id = ? AND job_seeker_id = ?
        ''', (job_id, current_user.id)).fetchone()
        if application:
            application_status = application['status']
            application_date = application['formatted_date']

    # Get number of applications for this job
    application_count = 0
    if current_user.is_authenticated and current_user.id == job['employer_id']:
        application_count = db.execute('''
            SELECT COUNT(*) as count
            FROM Applications
            WHERE job_id = ?
        ''', (job_id,)).fetchone()['count']
    
    return render_template('job_details.html',
                         job=job,
                         employer=job,
                         application_status=application_status,
                         application_date=application_date,
                         application_count=application_count,
                         has_applied=application_status is not None)

@app.route('/my_applications')
@login_required
def my_applications():
    if current_user.user_type != 'job_seeker':
        flash('Only job seekers can view applications.', 'warning')
        return redirect(url_for('dashboard'))
        
    db = get_db()
    
    # Get all applications with formatted dates
    applications = db.execute('''
        SELECT a.*, j.title, e.company_name,
               strftime('%Y-%m-%d', a.date_applied) as formatted_date
        FROM Applications a
        JOIN Jobs j ON a.job_id = j.job_id
        JOIN Users u ON j.employer_id = u.user_id
        JOIN Employer e ON u.user_id = e.user_id
        WHERE a.job_seeker_id = ?
        ORDER BY a.date_applied DESC
    ''', (current_user.id,)).fetchall()
    
    return render_template('my_applications.html', applications=applications)

# Flask application runner
if __name__ == '__main__':
    app.run(debug=True, port=5000)
