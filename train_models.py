from ml_models import JobPredictionModel, SentimentAnalysisModel
from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def connect_to_mongodb():
    try:
        client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017/'))
        db = client['job_prediction_db']
        # Test connection
        db.command('ping')
        logger.info("Successfully connected to MongoDB")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

def generate_sample_data(n_samples=1000):
    """Generate sample training data when real data is not available"""
    try:
        np.random.seed(42)
        
        # Generate job seeker features with realistic correlations
        data = {
            'total_applications': np.random.randint(1, 50, n_samples),
            'experience_years': np.random.uniform(0, 15, n_samples),
            'skills_count': np.random.randint(1, 20, n_samples),
            'average_job_salary': np.random.uniform(30000, 150000, n_samples)
        }
        
        # Calculate accepted applications with realistic correlations
        base_success_rate = 0.3
        experience_bonus = 0.05 * (data['experience_years'] / 15)
        skills_bonus = 0.05 * (data['skills_count'] / 20)
        success_rates = np.clip(base_success_rate + experience_bonus + skills_bonus, 0.1, 0.9)
        
        data['accepted_applications'] = np.random.binomial(
            data['total_applications'], 
            success_rates
        )
        data['success_rate'] = data['accepted_applications'] / data['total_applications']
        
        # Generate target (success probability) with more nuanced correlations
        success_prob = (
            0.3 +  # base probability
            0.3 * data['success_rate'] +  # past success weight
            0.2 * (data['experience_years'] / 15) +  # experience weight
            0.1 * (data['skills_count'] / 20) +  # skills weight
            0.1 * (data['average_job_salary'] - 30000) / 120000  # salary weight
        )
        success_prob = np.clip(success_prob, 0.1, 0.9)  # ensure reasonable bounds
        data['will_succeed'] = np.random.binomial(1, success_prob)
        
        logger.info(f"Generated {n_samples} sample records for job prediction training")
        return pd.DataFrame(data)
    
    except Exception as e:
        logger.error(f"Error generating sample data: {str(e)}")
        raise

def generate_sample_reviews(n_samples=1000):
    """Generate sample reviews for sentiment analysis with more variation and realism"""
    try:
        positive_templates = [
            "Great {role} with excellent {skill} skills. {positive_comment}",
            "Very professional and {positive_trait}. Demonstrated strong {skill} in {task}",
            "Highly recommended! {positive_comment} Excellent {skill} abilities",
            "Outstanding performance in {task}. {timeframe} of consistent excellence",
            "{timeframe} of excellent work. {positive_trait} and {positive_comment}"
        ]
        
        negative_templates = [
            "Poor {skill} skills and {negative_trait}. {issue}",
            "Not recommended due to {issue}. Needs improvement in {important_skill}",
            "Disappointing performance in {task}. {negative_trait}",
            "Lacks {important_skill}. {timeframe} of inconsistent work",
            "{timeframe} of subpar work. {issue} and {negative_trait}"
        ]
        
        # Expanded sample data for more variety
        roles = ['developer', 'manager', 'designer', 'analyst', 'engineer', 'consultant', 'coordinator']
        skills = ['communication', 'technical', 'leadership', 'problem-solving', 'teamwork', 'project management']
        positive_traits = ['reliable', 'innovative', 'dedicated', 'efficient', 'collaborative', 'proactive']
        negative_traits = ['unreliable', 'inflexible', 'unresponsive', 'inefficient', 'difficult', 'disorganized']
        tasks = ['project delivery', 'team management', 'problem solving', 'deadline management', 'client communication']
        timeframes = ['Six months', 'One year', 'Three months', 'Two years', 'Nine months']
        issues = [
            'missed deadlines', 'poor communication', 'lack of skills', 'unprofessional behavior',
            'inability to work in team', 'resistance to feedback'
        ]
        positive_comments = [
            'exceeded expectations', 'great team player', 'takes initiative', 'fast learner',
            'excellent problem solver', 'strong leadership skills'
        ]
        important_skills = [
            'time management', 'attention to detail', 'team coordination', 'technical expertise',
            'strategic planning', 'risk assessment'
        ]
        
        reviews = []
        sentiments = []
        
        for _ in range(n_samples):
            if np.random.random() > 0.3:  # 70% positive reviews
                template = np.random.choice(positive_templates)
                review = template.format(
                    role=np.random.choice(roles),
                    skill=np.random.choice(skills),
                    positive_trait=np.random.choice(positive_traits),
                    task=np.random.choice(tasks),
                    timeframe=np.random.choice(timeframes),
                    positive_comment=np.random.choice(positive_comments)
                )
                sentiment = 'positive'
            else:
                template = np.random.choice(negative_templates)
                review = template.format(
                    skill=np.random.choice(skills),
                    negative_trait=np.random.choice(negative_traits),
                    task=np.random.choice(tasks),
                    timeframe=np.random.choice(timeframes),
                    issue=np.random.choice(issues),
                    important_skill=np.random.choice(important_skills)
                )
                sentiment = 'negative'
            
            reviews.append(review)
            sentiments.append(sentiment)
        
        logger.info(f"Generated {n_samples} sample reviews for sentiment analysis training")
        return reviews, sentiments
    
    except Exception as e:
        logger.error(f"Error generating sample reviews: {str(e)}")
        raise

def prepare_training_data(db=None):
    """Prepare training data from either MongoDB or generate samples"""
    if db is None:
        print("No database connection. Using sample data...")
        return generate_sample_data(), generate_sample_reviews()
        
    # If db is available, get real data
    applications = list(db.applications.aggregate([
        {'$lookup': {
            'from': 'jobs',
            'localField': 'job_id',
            'foreignField': '_id',
            'as': 'job'
        }},
        {'$unwind': '$job'},
        {'$lookup': {
            'from': 'job_seeker_profiles',
            'localField': 'job_seeker_id',
            'foreignField': 'user_id',
            'as': 'profile'
        }},
        {'$unwind': '$profile'}
    ]))
    
    # Prepare features for each application
    data = []
    for app in applications:
        features = {
            'experience_years': float(app['profile'].get('experience_years', 0)),
            'skills_count': len(app['profile'].get('skills', '').split(',')) if app['profile'].get('skills') else 0,
            'job_salary': app['job'].get('salary', 0),
            'days_since_posted': (datetime.utcnow() - app['job']['date_posted']).days,
            'success': 1 if app['status'] == 'accepted' else 0
        }
        data.append(features)
    
    df = pd.DataFrame(data)
    
    if len(df) > 0:
        X = df.drop('success', axis=1)
        y = df['success']
        return X, y
    else:
        # Generate synthetic data if no real data exists
        return generate_synthetic_training_data()

def generate_synthetic_training_data(n_samples=1000):
    # Generate synthetic data for initial training
    np.random.seed(42)
    
    X = pd.DataFrame({
        'experience_years': np.random.uniform(0, 20, n_samples),
        'skills_count': np.random.randint(1, 15, n_samples),
        'job_salary': np.random.uniform(30000, 150000, n_samples),
        'days_since_posted': np.random.randint(1, 60, n_samples)
    })
    
    # Generate synthetic target based on some rules
    y = (0.3 * X['experience_years'] / 20 + 
         0.3 * X['skills_count'] / 15 + 
         0.2 * (X['job_salary'] - 30000) / 120000 + 
         0.2 * (60 - X['days_since_posted']) / 60 + 
         np.random.normal(0, 0.1, n_samples)) > 0.5
    y = y.astype(int)
    
    return X, y

def prepare_sentiment_training_data(db):
    # Get all reviews
    reviews = list(db.reviews.find())
    
    if len(reviews) > 0:
        texts = [review['review_text'] for review in reviews]
        sentiments = ['positive' if review['rating'] > 3 else 'negative' for review in reviews]
        return texts, sentiments
    else:
        # Generate synthetic reviews if no real data exists
        return generate_synthetic_reviews()

def generate_synthetic_reviews(n_samples=1000):
    positive_templates = [
        "Great job opportunity at {}! The {} role offers excellent {}.",
        "Amazing company culture at {}. The {} position provides wonderful {}.",
        "Excellent work environment at {}. The {} role has fantastic {}."
    ]
    
    negative_templates = [
        "Poor experience at {}. The {} position lacks {}.",
        "Disappointed with {} company. The {} role has inadequate {}.",
        "Unsatisfactory conditions at {}. The {} position needs better {}."
    ]
    
    companies = ["TechCorp", "DataSys", "InfoTech", "WebSolutions", "AICompany"]
    roles = ["Developer", "Analyst", "Manager", "Engineer", "Designer"]
    aspects = ["benefits", "growth opportunities", "work-life balance", "compensation", "team culture"]
    
    texts = []
    sentiments = []
    
    for _ in range(n_samples):
        if np.random.random() > 0.5:
            template = np.random.choice(positive_templates)
            sentiment = 'positive'
        else:
            template = np.random.choice(negative_templates)
            sentiment = 'negative'
        
        review = template.format(
            np.random.choice(companies),
            np.random.choice(roles),
            np.random.choice(aspects)
        )
        
        texts.append(review)
        sentiments.append(sentiment)
    
    return texts, sentiments

def train_and_evaluate_models(n_samples=1000):
    """Train both models and evaluate their performance"""
    try:
        # Initialize models
        job_model = JobPredictionModel()
        sentiment_model = SentimentAnalysisModel()
        
        # Generate and prepare job prediction data
        logger.info("Generating job prediction training data...")
        job_data = generate_sample_data(n_samples)
        X = job_data.drop('will_succeed', axis=1)
        y = job_data['will_succeed']
        
        # Train job prediction model
        logger.info("Training job prediction model...")
        job_metrics = job_model.train(X, y)
        logger.info(f"Job prediction model metrics: {job_metrics}")
        
        # Generate and prepare sentiment analysis data
        logger.info("Generating sentiment analysis training data...")
        reviews, sentiments = generate_sample_reviews(n_samples)
        
        # Train sentiment analysis model
        logger.info("Training sentiment analysis model...")
        sentiment_metrics = sentiment_model.train(reviews, sentiments)
        logger.info(f"Sentiment analysis model metrics: {sentiment_metrics}")
        
        return {
            'job_prediction_metrics': job_metrics,
            'sentiment_analysis_metrics': sentiment_metrics
        }
        
    except Exception as e:
        logger.error(f"Error in model training and evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        metrics = train_and_evaluate_models(n_samples=2000)
        logger.info("Model training completed successfully")
        logger.info("Job Prediction Metrics:")
        for k, v in metrics['job_prediction_metrics'].items():
            logger.info(f"{k}: {v}")
        logger.info("\nSentiment Analysis Metrics:")
        for k, v in metrics['sentiment_analysis_metrics'].items():
            if k != 'confusion_matrix':
                logger.info(f"{k}: {v}")
    except Exception as e:
        logger.error(f"Training script failed: {str(e)}")
