import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import json
from datetime import datetime
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metrics_path = os.path.join(os.getcwd(), 'models', f'{model_name}_metrics.json')
        self.history = self._load_history()

    def _load_history(self):
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error loading metrics history for {self.model_name}")
                return {'metrics': [], 'predictions': []}
        return {'metrics': [], 'predictions': []}

    def log_metrics(self, metrics):
        metrics['timestamp'] = datetime.now().isoformat()
        self.history['metrics'].append(metrics)
        self._save_history()

    def log_prediction(self, features, prediction, actual=None):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction': float(prediction),
            'features': features.to_dict('records')[0] if isinstance(features, pd.DataFrame) else features,
        }
        if actual is not None:
            log_entry['actual'] = actual
        self.history['predictions'].append(log_entry)
        self._save_history()

    def _save_history(self):
        try:
            with open(self.metrics_path, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            logger.error(f"Error saving metrics history for {self.model_name}: {str(e)}")

class JobPredictionModel:
    def __init__(self):
        workspace_dir = os.getcwd()
        models_dir = os.path.join(workspace_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
            
        self.model_path = os.path.join(models_dir, 'job_prediction_model.joblib')
        self.scaler_path = os.path.join(models_dir, 'scaler.joblib')
        self.model = None
        self.scaler = None
        self.monitor = ModelMonitor('job_prediction')
        
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Loaded existing job prediction model and scaler")
            else:
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                self.scaler = StandardScaler()
                logger.info("Initialized new job prediction model")
        except Exception as e:
            logger.error(f"Error loading job prediction model: {str(e)}")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()

    def prepare_features(self, applications_data, user_profile):
        try:
            features = {
                'total_applications': len(applications_data),
                'accepted_applications': sum(1 for app in applications_data if app.get('status') == 'accepted'),
                'average_job_salary': np.mean([app.get('job_salary', 0) for app in applications_data]) if applications_data else 0,
                'experience_years': float(user_profile.get('experience_years', 0)),
                'skills_count': len(user_profile.get('skills', '').split(',')) if user_profile.get('skills') else 0
            }
            
            if features['total_applications'] > 0:
                features['success_rate'] = features['accepted_applications'] / features['total_applications']
            else:
                features['success_rate'] = 0
                
            return pd.DataFrame([features])
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise ValueError("Failed to prepare features for prediction")

    def train(self, X, y):
        try:
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Calculate metrics
            y_pred = self.model.predict(X_val_scaled)
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'cv_scores': cross_val_score(self.model, X_train_scaled, y_train, cv=5).tolist()
            }
            
            # Log metrics
            self.monitor.log_metrics(metrics)
            
            # Save model and scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            logger.info(f"Model trained successfully. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise RuntimeError("Failed to train the model")

    def predict_success_probability(self, features_df):
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            X_scaled = self.scaler.transform(features_df)
            prediction = self.model.predict_proba(X_scaled)[:, 1][0]
            
            # Log prediction
            self.monitor.log_prediction(features_df, prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise RuntimeError("Failed to make prediction")

class SentimentAnalysisModel:
    def __init__(self):
        workspace_dir = os.getcwd()
        models_dir = os.path.join(workspace_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
            
        self.model_path = os.path.join(models_dir, 'sentiment_model.joblib')
        self.vectorizer_path = os.path.join(models_dir, 'vectorizer.joblib')
        self.model = None
        self.vectorizer = None
        self.monitor = ModelMonitor('sentiment_analysis')
        
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                logger.info("Loaded existing sentiment analysis model and vectorizer")
            else:
                self.model = MultinomialNB()
                self.vectorizer = TfidfVectorizer(max_features=5000)
                logger.info("Initialized new sentiment analysis model")
        except Exception as e:
            logger.error(f"Error loading sentiment analysis model: {str(e)}")
            self.model = MultinomialNB()
            self.vectorizer = TfidfVectorizer(max_features=5000)

    def train(self, reviews, sentiments):
        try:
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(reviews, sentiments, test_size=0.2, random_state=42)
            
            # Transform text data
            X_train_vectorized = self.vectorizer.fit_transform(X_train)
            X_val_vectorized = self.vectorizer.transform(X_val)
            
            # Train model
            self.model.fit(X_train_vectorized, y_train)
            
            # Calculate metrics
            y_pred = self.model.predict(X_val_vectorized)
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='weighted'),
                'recall': recall_score(y_val, y_pred, average='weighted'),
                'f1': f1_score(y_val, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
            }
            
            # Log metrics
            self.monitor.log_metrics(metrics)
            
            # Save model and vectorizer
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.vectorizer, self.vectorizer_path)
            
            logger.info(f"Sentiment model trained successfully. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training sentiment model: {str(e)}")
            raise RuntimeError("Failed to train the sentiment analysis model")

    def predict_sentiment(self, review_text):
        try:
            if self.model is None or self.vectorizer is None:
                raise ValueError("Model not trained yet")
            
            # Handle single string or list of strings
            if isinstance(review_text, str):
                review_text = [review_text]
            
            X = self.vectorizer.transform(review_text)
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            # Log predictions
            for i, (text, pred) in enumerate(zip(review_text, predictions)):
                self.monitor.log_prediction({'text': text}, pred)
            
            return predictions[0] if len(review_text) == 1 else predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error predicting sentiment: {str(e)}")
            raise RuntimeError("Failed to predict sentiment")

class CoverLetterEvaluationModel:
    def __init__(self):
        workspace_dir = os.getcwd()
        models_dir = os.path.join(workspace_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        self.model_path = os.path.join(models_dir, 'cover_letter_model')
        self.monitor = ModelMonitor('cover_letter_evaluation')
        
        self.feedback_path = os.path.join(models_dir, 'evaluation_feedback.json')
        self.load_feedback_history()
    
        try:
            # Load SBERT model for semantic similarity
            self.encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            
            # Load spaCy for NLP tasks
            self.nlp = spacy.load('en_core_web_sm')
            
            logger.info("Initialized cover letter evaluation model")
        except Exception as e:
            logger.error(f"Error initializing cover letter evaluation model: {str(e)}")
            raise
    
    def load_feedback_history(self):
        """Load historical feedback data"""
        try:
            if os.path.exists(self.feedback_path):
                with open(self.feedback_path, 'r') as f:
                    self.feedback_history = json.load(f)
            else:
                self.feedback_history = []
            logger.info(f"Loaded {len(self.feedback_history)} feedback entries")
        except Exception as e:
            logger.error(f"Error loading feedback history: {str(e)}")
            self.feedback_history = []
    
    def save_feedback(self, feedback_data):
        """Save feedback and update weights"""
        try:
            feedback_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'evaluation_id': feedback_data['evaluation_id'],
                'original_score': feedback_data['original_score'],
                'feedback_score': feedback_data['feedback_score'],
                'recruiter_rating': feedback_data.get('recruiter_rating'),
                'comments': feedback_data.get('comments', '')
            }
            
            self.feedback_history.append(feedback_entry)
            
            # Save updated feedback history
            with open(self.feedback_path, 'w') as f:
                json.dump(self.feedback_history, f)
            
            # Adjust weights based on feedback
            self._update_weights()
            
            logger.info(f"Saved feedback for evaluation {feedback_data['evaluation_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
            return False
    
    def _update_weights(self):
        """Update model weights based on feedback history"""
        try:
            if len(self.feedback_history) < 10:  # Need minimum feedback to adjust
                return
            
            # Calculate average difference between original and feedback scores
            score_diffs = []
            for entry in self.feedback_history[-50:]:  # Use last 50 entries
                if entry['original_score'] and entry['feedback_score']:
                    diff = entry['feedback_score'] - entry['original_score']
                    score_diffs.append(diff)
            
            if score_diffs:
                avg_diff = sum(score_diffs) / len(score_diffs)
                
                # Adjust weights based on feedback trend
                if abs(avg_diff) > 5:  # Only adjust if difference is significant
                    if avg_diff > 0:  # Model scoring too low
                        self.semantic_weight = min(0.8, self.semantic_weight + 0.05)
                        self.skills_weight = 1 - self.semantic_weight
                    else:  # Model scoring too high
                        self.semantic_weight = max(0.6, self.semantic_weight - 0.05)
                        self.skills_weight = 1 - self.semantic_weight
                
                logger.info(f"Updated weights: semantic={self.semantic_weight:.2f}, skills={self.skills_weight:.2f}")
        
        except Exception as e:
            logger.error(f"Error updating weights: {str(e)}")

    def preprocess_text(self, text):
        """Clean and normalize text"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and extra whitespace
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            # Process with spaCy
            doc = self.nlp(text)
            
            # Remove stopwords and lemmatize
            processed = ' '.join([token.lemma_ for token in doc if not token.is_stop])
            
            return processed
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text
    
    def extract_skills(self, text):
        """Extract skills from text using spaCy NER and pattern matching"""
        skills = set()
        doc = self.nlp(text)
        
        # Add named entities that might be skills
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT']:
                skills.add(ent.text.lower())
        
        # Add noun chunks that might be skills
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to 3-word phrases
                skills.add(chunk.text.lower())
        
        return list(skills)
    
    def evaluate_cover_letter(self, job_description, cover_letter, required_skills=None):
        """
        Evaluate cover letter against job description
        Returns: dict with score and analysis
        """
        try:
            # Preprocess texts
            proc_job = self.preprocess_text(job_description)
            proc_cover = self.preprocess_text(cover_letter)
            
            # Get embeddings
            job_embedding = self.encoder.encode([proc_job])[0]
            cover_embedding = self.encoder.encode([proc_cover])[0]
            
            # Calculate semantic similarity
            similarity = cosine_similarity(
                job_embedding.reshape(1, -1),
                cover_embedding.reshape(1, -1)
            )[0][0]
            
            # Extract skills
            job_skills = set(self.extract_skills(job_description))
            cover_skills = set(self.extract_skills(cover_letter))
            
            if required_skills:
                job_skills.update(set(skill.lower() for skill in required_skills))
            
            # Calculate skills match
            matched_skills = job_skills.intersection(cover_skills)
            skills_score = len(matched_skills) / len(job_skills) if job_skills else 0
            
            # Use dynamic weights from feedback
            semantic_weight = getattr(self, 'semantic_weight', 0.7)
            skills_weight = getattr(self, 'skills_weight', 0.3)
            
            # Calculate final score with dynamic weights
            final_score = (semantic_weight * similarity + skills_weight * skills_score) * 100
            
            # Generate unique evaluation ID
            evaluation_id = str(ObjectId())
            
            # Prepare analysis
            analysis = {
                'evaluation_id': evaluation_id,
                'semantic_similarity': round(similarity * 100, 2),
                'skills_match_percentage': round(skills_score * 100, 2),
                'matched_skills': list(matched_skills),
                'missing_skills': list(job_skills - cover_skills),
                'overall_score': round(final_score, 2),
                'weights_used': {
                    'semantic': semantic_weight,
                    'skills': skills_weight
                }
            }
            
            # Log prediction
            self.monitor.log_prediction(
                {'job_len': len(proc_job), 'cover_len': len(proc_cover)},
                analysis['overall_score']
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error evaluating cover letter: {str(e)}")
            raise
