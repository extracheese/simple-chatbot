"""Main Flask application file for the chatbot."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file *first*
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
loaded = load_dotenv(dotenv_path=dotenv_path)
if loaded:
    print(f"Loaded environment variables from: {dotenv_path}") # Simple print for initial check
else:
    print(f"Warning: .env file not found or not loaded from: {dotenv_path}")

# Standard library imports
import logging
import os
from flask import Flask, request, jsonify, render_template, session
import uuid
import sys
from chatbot.logging_config import setup_logging
setup_logging() # Call setup immediately

# Now get loggers, they are configured
startup_logger = logging.getLogger('startup')
chat_logger = logging.getLogger('chat')
menu_logger = logging.getLogger('menu.processor')

from chatbot.state_manager import (
    get_or_create_order,
    get_or_create_memory,
    update_memory,
    load_order_from_dict,
    get_memory
)
from chatbot.tools import search_menu, update_order_tool, confirm_order, tools_schema
from chatbot.orchestrator import generate_response
from menu.processor import MenuProcessor, MenuItem 
import config

# --- Validate Configuration --- #
if not config.validate_config():
    startup_logger.critical("Essential configuration missing. Exiting application.")
    sys.exit("Exiting due to missing configuration.")
else:
    startup_logger.info("Configuration validated successfully.")

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = config.FLASK_SECRET_KEY

# --- Component Initialization ---
# Load Menu Data
menu_file = config.MENU_FILE_PATH

# Create MenuProcessor instance
processor = MenuProcessor()

# Load the menu data using the process_menu method
try:
    processor.process_menu(menu_file)
    startup_logger.info(f"Successfully processed menu file: {menu_file}")
except FileNotFoundError:
    startup_logger.error(f"CRITICAL ERROR: Menu file not found during processing: {menu_file}")
    sys.exit("Exiting due to missing menu file.")
except Exception as e:
    startup_logger.error(f"CRITICAL ERROR: Failed to process menu file {menu_file}: {e}")
    sys.exit("Exiting due to menu processing error.")

# --- Model Initialization (To be moved to llm_client.py later) ---
startup_logger.info("LLM client initialization delegated to chatbot.llm_client.")

# === API Endpoints === #

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    message = data.get('message')
    session_id = data.get('session_id')

    if not message or not session_id:
        chat_logger.error(f"Missing message or session_id in request: {data}")
        return jsonify({"error": "Message and session_id are required"}), 400

    # Retrieve the current order state (memory is handled within orchestrator)
    current_order = get_or_create_order(session_id)

    chat_logger.info(f"Session {session_id} - Received message: '{message}'")
    chat_logger.debug(f"Session {session_id} - Order state before processing: {current_order.to_dict()}")

    try:
        # Generate response using the imported orchestrator function
        response_data = generate_response(
            session_id=session_id,
            user_message=message,
            current_order=current_order, # Pass the current order state
            processor=processor # Pass the menu processor
        )

        if response_data:
            # The order object within response_data *should* be the mutated one,
            # but let's fetch explicitly from state_manager to be safe, 
            # as orchestrator modifies the passed 'current_order' object directly.
            # updated_order_state = get_or_create_order(session_id).to_dict()
            # response_data['order'] = updated_order_state 
            # ^-- Re-evaluate if this explicit fetch is needed. If orchestrator correctly modifies
            #     the passed order object, response_data['order'] should be up-to-date.

            chat_logger.info(f"Session {session_id} - Sending response: {response_data['response'][:100]}... Order state: {response_data['order']['status']}")
            return jsonify(response_data)

    except Exception as e:
        chat_logger.error(f"Session {session_id} - Unexpected error during message processing: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred while processing the message."}), 500

@app.route('/')
def index():
    """Serves the main HTML page."""
    # Generate a unique session ID if one doesn't exist
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/update_order', methods=['POST'])
def update_order():
    data = request.json
    session_id = data.get('session_id')

    try:
        # Use the state manager function to update the order based on request data
        load_order_from_dict(session_id, data, processor) # Pass processor for validation within load
        
        # Fetch the potentially updated order object
        updated_order = get_or_create_order(session_id)
        
        chat_logger.info(f"Session {session_id} - Order state AFTER update: {updated_order.to_dict()}")
        return jsonify(updated_order.to_dict())
    
    except ValueError as ve:
        chat_logger.error(f"Session {session_id} - Error updating order: {ve}", exc_info=True)
        return jsonify({"error": str(ve)}), 400 # Return validation errors as Bad Request
    except Exception as e:
        chat_logger.error(f"Session {session_id} - Unexpected error during order update: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred while updating the order."}), 500

@app.route('/get_order/<session_id>', methods=['GET'])
def get_order(session_id):
    # Retrieve the current order and memory state
    current_order = get_or_create_order(session_id)
    memory = get_or_create_memory(session_id)

    chat_logger.info(f"Session {session_id} - Retrieved order: {current_order.to_dict()}")
    return jsonify(current_order.to_dict())

# --- Main Execution Guard --- #
if __name__ == '__main__':
    # This block runs only when script is executed directly (e.g., `python application.py`)
    # For production, use a WSGI server like Gunicorn (see MEMORY/deployment notes)
    startup_logger.info("Starting Flask development server...")
    # Ensure log directory exists before starting
    # Use LOG_DIRECTORY from config
    app.run(debug=True, host='0.0.0.0', port=5000)
