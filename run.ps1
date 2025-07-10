# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Initialize the database and train models
python train_models.py

# Run the application
$env:FLASK_ENV = "development"
$env:FLASK_DEBUG = "1"
python app1.py
