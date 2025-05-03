#!/bin/bash
# Script to run the chatbot application with a clean environment
# This ensures that the application always starts with fresh environment variables

echo "Starting MickeyDs Chatbot application..."
echo "Activating virtual environment..."
source venv/bin/activate

echo "Running application with Gunicorn..."
gunicorn --workers 1 --bind 0.0.0.0:8000 application:app --log-level debug --access-logfile - --error-logfile -

# If you want to run with Flask's development server instead (uncomment the line below)
# python application.py
