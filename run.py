import os
from app1 import app
import os

def ensure_directories():
    """Ensure required directories exist"""
    directories = ['static', 'uploads']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

if __name__ == '__main__':
    try:
        print("Starting Flask application...")
        ensure_directories()
        print("Server will be available at http://localhost:5000")
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Error starting server: {e}")
        exit(1)
