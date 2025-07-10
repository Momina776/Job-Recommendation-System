# Stop any existing MongoDB service
$mongoService = Get-Service -Name "MongoDB" -ErrorAction SilentlyContinue
if ($mongoService -and $mongoService.Status -eq "Running") {
    Write-Host "MongoDB is already running"
} else {
    Write-Host "Starting MongoDB..."
    Start-Service -Name "MongoDB"
    Start-Sleep -Seconds 5
}

# Check if virtual environment exists
if (-not (Test-Path ".\venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install requirements
Write-Host "Installing requirements..."
pip install -r requirements.txt

# Initialize ML models
Write-Host "Initializing ML models..."
python train_models.py

# Start the Flask application
Write-Host "Starting Flask application..."
$env:FLASK_ENV = "development"
$env:FLASK_DEBUG = "1"
python app1.py
