#!/usr/bin/env python
"""
Script to reset environment variables and start the application.
This ensures that any cached environment variables are cleared before starting.
"""
import os
import sys
import subprocess

# Clear any existing Azure OpenAI environment variables
azure_vars = [
    'AZURE_OPENAI_DEPLOYMENT_NAME',
    'AZURE_OPENAI_MODEL',
    'AZURE_API_VERSION',
    'AZURE_OPENAI_ENDPOINT',
    'AZURE_OPENAI_API_KEY'
]

for var in azure_vars:
    if var in os.environ:
        print(f"Removing existing {var}")
        del os.environ[var]

# Start Gunicorn with a clean environment
cmd = [
    "gunicorn",
    "--workers", "4",
    "--bind", "0.0.0.0:8000",
    "application:app",
    "--log-level", "debug",
    "--access-logfile", "-",
    "--error-logfile", "-"
]

print(f"Starting Gunicorn with command: {' '.join(cmd)}")
subprocess.run(cmd)
