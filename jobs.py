from datetime import datetime, date
from database.db_connection import db

class Job(db.Model):
    __tablename__ = 'jobs'
    
    job_id = db.Column(db.Integer, primary_key=True)
    employer_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    salary = db.Column(db.Numeric(10, 2))
    deadline = db.Column(db.Date)
    status = db.Column(db.Enum('open', 'closed', 'hiring', 'cancelled'), default='open')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    applications = db.relationship('Application', backref='job', lazy=True)