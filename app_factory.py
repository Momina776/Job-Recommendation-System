import os
import sys
from flask import Flask
from config import Config
from db_utils import get_db_connection
from flask_login import LoginManager

def create_app(config_class=Config):
    # Initialize Flask app
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Ensure upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    
    # Initialize database connection
    try:
        client, error = get_db_connection()
        if error:
            print(f"Error connecting to database: {error}")
            sys.exit(1)
        
        # Store MongoDB client in app config
        app.config['mongo_client'] = client
        app.config['db'] = client[Config.MONGO_DB_NAME]
        print("Successfully connected to MongoDB!")
        
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        sys.exit(1)
    
    # Register error handlers
    from error_handlers import register_error_handlers
    register_error_handlers(app)
    
    # Register blueprints
    from routes.auth import auth_bp
    from routes.jobs import jobs_bp
    from routes.applications import applications_bp
    from routes.profiles import profiles_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(jobs_bp)
    app.register_blueprint(applications_bp)
    app.register_blueprint(profiles_bp)
    
    return app
