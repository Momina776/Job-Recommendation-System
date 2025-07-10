from flask import Flask, render_template, request, redirect, url_for, flash, session, g
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

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

# --- Flask-Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

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
    return render_template('index.html')

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
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        user_type = request.form.get('user_type')

        # Ensure name is split into first_name and last_name
        if name:
            name_parts = name.split(' ', 1)
            first_name = name_parts[0]
            last_name = name_parts[1] if len(name_parts) > 1 else ''
        else:
            first_name = ''
            last_name = ''

        # Validate required fields
        if not first_name or not email or not password or not user_type:
            flash('Please fill out all required fields.', 'danger')
            return redirect(url_for('register'))

        # Insert user into the database
        db = get_db()
        db.execute(
            'INSERT INTO Users (first_name, last_name, email, password, user_type) '
            'VALUES (?, ?, ?, ?, ?)',
            (first_name, last_name, email, password, user_type)
        )
        db.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/jobs')
def jobs():
    db = get_db()
    cur = db.execute('SELECT * FROM Jobs')
    jobs = cur.fetchall()
    return render_template('jobs.html', jobs=jobs)

@app.route('/job/<int:job_id>')
def job_details(job_id):
    db = get_db()
    cur = db.execute('SELECT * FROM Jobs WHERE job_id = ?', (job_id,))
    job = cur.fetchone()
    if job is None:
        flash('Job not found.', 'danger')
        return redirect(url_for('jobs'))
    return render_template('job_details.html', job=job)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.name)

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@app.route('/my_applications')
@login_required
def my_applications():
    # Add your logic here
    return render_template('my_applications.html')

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)


'''''
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Mock user class for demonstration
class User(UserMixin):
    def __init__(self, user_id, name, email, user_type):
        self.id = user_id
        self.name = name
        self.email = email
        self.user_type = user_type

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    # In a real app, you would query your database here
    return User(user_id, "Test User", "test@example.com", "job_seeker")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # In a real app, you would verify credentials here
        user = User(1, "Test User", request.form.get('email'), "job_seeker")
        login_user(user)
        flash('Logged in successfully!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # In a real app, you would create a new user here
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/jobs')
def jobs():
    # Mock jobs data
    jobs = [
        {
            'job_id': 1,
            'title': 'Senior Software Developer',
            'company': 'TechCorp',
            'description': 'We are looking for an experienced developer...',
            'salary': '$90,000 - $120,000',
            'deadline': '2023-12-31',
            'employer_id': 1
        },
        # Add more mock jobs as needed
    ]
    return render_template('jobs.html', jobs=jobs)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/profile')
@login_required  
def profile():
    return render_template('profile.html')
    
if __name__ == '__main__':
    app.run(debug=True)
   
'''

