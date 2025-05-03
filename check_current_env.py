#!/usr/bin/env python3
import os
import json
from dotenv import load_dotenv

# Print all current Azure OpenAI related environment variables
print("CURRENT ENVIRONMENT VARIABLES:")
azure_vars = {k: v for k, v in os.environ.items() if k.startswith('AZURE_')}
print(json.dumps(azure_vars, indent=2))

# Clear the variables
print("\nCLEARING VARIABLES...")
for var in ['AZURE_OPENAI_DEPLOYMENT_NAME', 'AZURE_OPENAI_MODEL', 'AZURE_API_VERSION', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY']:
    if var in os.environ:
        print(f"Removing {var}")
        del os.environ[var]
    else:
        print(f"{var} not found in environment")

# Print after clearing
print("\nAFTER CLEARING:")
azure_vars = {k: v for k, v in os.environ.items() if k.startswith('AZURE_')}
print(json.dumps(azure_vars, indent=2))

# Load from .env file
print("\nLOADING FROM .ENV FILE...")
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
loaded = load_dotenv(dotenv_path=dotenv_path, override=True)
print(f"Loaded: {loaded} from {dotenv_path}")

# Print after loading
print("\nAFTER LOADING FROM .ENV:")
azure_vars = {k: v for k, v in os.environ.items() if k.startswith('AZURE_')}
print(json.dumps(azure_vars, indent=2))

# Print specific variables of interest
print("\nSpecific variables:")
print(f"AZURE_OPENAI_DEPLOYMENT_NAME: {os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME')}")
print(f"AZURE_OPENAI_MODEL: {os.environ.get('AZURE_OPENAI_MODEL')}")
