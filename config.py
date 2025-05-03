import os
import logging

# Get logger instance (ensure logging is configured before use)
# Logging setup is expected to be done before this module is imported
startup_logger = logging.getLogger('startup')

# --- Environment Variable Loading --- #
# Variables are now expected to be loaded into os.environ *before* this module is imported.

def get_env_variable(var_name: str, is_secret: bool = True) -> str | None:
    """Gets an environment variable, logging appropriate messages."""
    value = os.environ.get(var_name)
    # --- Cascade Debug --- Add logging for the raw value retrieved for the specific key
    if var_name == "AZURE_OPENAI_DEPLOYMENT_NAME":
        startup_logger.debug(f"Raw value for {var_name} from os.environ.get: '{value}' (Type: {type(value)})")
    # --- End Cascade Debug ---
    if value:
        if is_secret:
            startup_logger.info(f"Loaded environment variable: {var_name}")
        else:
            startup_logger.info(f"Loaded environment variable: {var_name} = {value}")
    else:
        startup_logger.warning(f"Environment variable not set: {var_name}")
    return value

# Azure OpenAI Credentials
AZURE_OPENAI_ENDPOINT = get_env_variable("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = get_env_variable("AZURE_OPENAI_API_KEY")
# Load API version from environment or use a default
AZURE_API_VERSION = get_env_variable("AZURE_API_VERSION", is_secret=False) or "2024-02-01"
startup_logger.info(f"Using Azure API version: {AZURE_API_VERSION}")

# Flask Configuration
FLASK_SECRET_KEY = get_env_variable("FLASK_SECRET_KEY") or 'a_default_secret_key_for_development'

# --- Application Constants --- #

# Get the directory where config.py is located
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# File Paths
MENU_FILE_PATH = os.path.join(CONFIG_DIR, 'data', 'mcdonalds_fixed_full.json') # Use absolute path
LOG_DIRECTORY = 'logs'

# Models
# Read deployment name from environment, fall back to "gpt-4" if not set
AZURE_OPENAI_DEPLOYMENT = get_env_variable("AZURE_OPENAI_DEPLOYMENT_NAME", is_secret=False) or "bad deployment name" 
startup_logger.info(f"Using Azure deployment name: {AZURE_OPENAI_DEPLOYMENT}")

# Read model name from environment, fall back to deployment name if not set
AZURE_OPENAI_MODEL = get_env_variable("AZURE_OPENAI_MODEL", is_secret=False) or "bad model name"
startup_logger.info(f"Using Azure model name: {AZURE_OPENAI_MODEL}")


# Logging
LOG_FILE_NAME = 'debug.log'
LOG_MAX_BYTES = 5 * 1024 * 1024 # 5 MB
LOG_BACKUP_COUNT = 3

# --- Validation --- #

def validate_config():
    """Checks if essential configuration variables are set."""
    essential_vars = {
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "FLASK_SECRET_KEY": FLASK_SECRET_KEY
    }
    all_set = True
    for name, value in essential_vars.items():
        if not value:
            startup_logger.error(f"CRITICAL CONFIG ERROR: Environment variable {name} is not set.")
            all_set = False
    
    if not os.path.exists(MENU_FILE_PATH):
        startup_logger.error(f"CRITICAL CONFIG ERROR: Menu file not found at {MENU_FILE_PATH}")
        all_set = False
        
    return all_set

# Example of how to use:
# from config import AZURE_OPENAI_ENDPOINT, MENU_FILE_PATH, validate_config
# if not validate_config():
#     sys.exit("Exiting due to missing configuration.")
