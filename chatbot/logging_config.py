import logging
import os
from logging.handlers import RotatingFileHandler

log_directory = "logs"

def setup_logging():
    """Configures logging for the application."""
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Basic configuration (adjust levels and handlers as needed)
    # Avoid configuring root logger here if specific handlers are used
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') 

    # --- Rotating File Handler for Chat Logs ---
    chat_log_file = os.path.join(log_directory, 'debug.log')
    # Rotate logs at 5MB, keep 3 backup files
    file_handler = RotatingFileHandler(chat_log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) # Log DEBUG level and above to file
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # --- Get Specific Loggers ---
    startup_logger = logging.getLogger('startup')
    chat_logger = logging.getLogger('chat')
    # Update logger name to reflect the new module path
    menu_logger = logging.getLogger('menu.processor') 

    # --- Configure Specific Loggers ---
    # Add file handler to relevant loggers
    chat_logger.addHandler(file_handler)
    chat_logger.setLevel(logging.DEBUG) # Ensure chat logger captures DEBUG
    startup_logger.addHandler(file_handler)
    startup_logger.setLevel(logging.DEBUG) # Changed from INFO to DEBUG
    menu_logger.addHandler(file_handler)
    menu_logger.setLevel(logging.DEBUG) # Changed from INFO to DEBUG

    # Prevent loggers from propagating to the root logger 
    chat_logger.propagate = False
    startup_logger.propagate = False
    menu_logger.propagate = False

    startup_logger.info("Logging configured successfully.")

    # Loggers are typically accessed globally via logging.getLogger after setup,
    # so no need to return them unless specifically required.
