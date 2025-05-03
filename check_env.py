#!/usr/bin/env python
"""Simple script to check environment variables loaded from .env file"""
import os
from dotenv import load_dotenv

# First, clear any existing environment variables
if 'AZURE_OPENAI_DEPLOYMENT_NAME' in os.environ:
    print(f"Removing existing AZURE_OPENAI_DEPLOYMENT_NAME: {os.environ['AZURE_OPENAI_DEPLOYMENT_NAME']}")
    del os.environ['AZURE_OPENAI_DEPLOYMENT_NAME']

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

# First, let's print the raw content of the .env file
print("Raw content of .env file:")
try:
    with open(dotenv_path, 'r') as f:
        content = f.read()
        print(content)
except Exception as e:
    print(f"Error reading .env file: {e}")

# Now load the environment variables
loaded = load_dotenv(dotenv_path=dotenv_path, override=True)
if loaded:
    print(f"\nLoaded environment variables from: {dotenv_path}")
else:
    print(f"\nWarning: .env file not found or not loaded from: {dotenv_path}")

# Print out the key environment variables
print("\nAzure OpenAI Configuration:")
print(f"AZURE_OPENAI_ENDPOINT: {os.environ.get('AZURE_OPENAI_ENDPOINT')}")
print(f"AZURE_OPENAI_API_KEY: {'*****' if os.environ.get('AZURE_OPENAI_API_KEY') else 'Not set'}")
print(f"AZURE_OPENAI_DEPLOYMENT_NAME: {os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME')}")
print(f"AZURE_OPENAI_MODEL: {os.environ.get('AZURE_OPENAI_MODEL')}")
print(f"AZURE_API_VERSION: {os.environ.get('AZURE_API_VERSION')}")
