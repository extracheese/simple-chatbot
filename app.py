from flask import Flask, render_template, request, jsonify
from openai import AzureOpenAI
import google.generativeai as genai
from dotenv import load_dotenv
import os
from menu_processor import processor
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
from enum import Enum
import re
import logging
from order_update_schema import order_update_schema
from langchain.memory import ConversationBufferMemory

# Set up chat conversation logging
chat_log_path = os.path.join(os.path.dirname(__file__), "chat_conversation.log")
chat_logger = logging.getLogger("chat_logger")
chat_logger.setLevel(logging.INFO)
if not chat_logger.hasHandlers():
    handler = logging.FileHandler(chat_log_path, encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    chat_logger.addHandler(handler)

# Load the system prompt from an external file
with open("system_prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

class ItemStatus(Enum):
    CONFIRMED = "confirmed"
    NEEDS_CUSTOMIZATION = "needs_customization"

@dataclass
class OrderItem:
    id: str
    name: str
    price: float
    quantity: int = 1
    customizations: Dict[str, str] = None
    status: ItemStatus = ItemStatus.CONFIRMED
    clarification_needed: Optional[str] = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "price": self.price,
            "quantity": self.quantity,
            "customizations": self.customizations or {},
            "status": self.status.value,
            "clarification_needed": self.clarification_needed
        }

@dataclass
class Order:
    items: List[OrderItem] = field(default_factory=list)
    total: float = 0
    current_clarification_index: int = 0

    def add_item(self, item_id: str, name: str, price: float, status: ItemStatus = ItemStatus.CONFIRMED, 
                 clarification_needed: str = None, customizations: Dict[str, str] = None):
        # VALIDATE: Ensure item exists in menu
        menu_item = processor.get_menu_item(item_id)
        if not menu_item:
            raise ValueError(f"Item with id '{item_id}' does not exist in the menu.")
        # VALIDATE: Ensure name and price match the menu
        if name != menu_item.name:
            raise ValueError(f"Name mismatch for item id '{item_id}': got '{name}', expected '{menu_item.name}'")
        if float(price) != float(menu_item.price):
            raise ValueError(f"Price mismatch for item id '{item_id}': got '{price}', expected '{menu_item.price}'")

        # Start with defaults
        final_customizations = menu_item.get_default_customizations()
        
        # Override with user-specified customizations
        if customizations:
            final_customizations.update(customizations)
        
        # Check if any customization options are missing and need clarification
        if menu_item.customization_options:
            for option_type, options_data in menu_item.customization_options.items():
                if option_type not in final_customizations:
                    # If this option has no default, we need clarification
                    if "default" not in options_data and "defaults" not in options_data:
                        status = ItemStatus.NEEDS_CUSTOMIZATION
                        clarification_needed = f"Please choose a {option_type} option: {', '.join(str(opt) for opt in options_data['options'])}"
                        break
        
        # Convert string status to enum if needed
        if isinstance(status, str):
            status = ItemStatus(status)
        
        new_item = OrderItem(
            id=item_id, 
            name=name, 
            price=price, 
            customizations=final_customizations,
            status=status,
            clarification_needed=clarification_needed
        )
        self.items.append(new_item)
        self.calculate_total()
        return new_item

    def remove_item(self, index: int):
        if 0 <= index < len(self.items):
            self.items.pop(index)
            self.calculate_total()
            return True
        return False

    def update_item_status(self, item_id: str, status: ItemStatus, price: Optional[float] = None):
        """Update the status of an item"""
        for item in self.items:
            if item.id == item_id:
                if price is not None:
                    item.price = price
                item.status = status
                self.calculate_total()
                return True
        return False

    def get_next_clarification(self) -> Optional[tuple[int, OrderItem]]:
        """Get the next item that needs clarification"""
        for i, item in enumerate(self.items[self.current_clarification_index:], self.current_clarification_index):
            if item.status != ItemStatus.CONFIRMED:
                return i, item
        return None

    def update_next_clarification(self, price: float, customizations: Dict[str, str] = None):
        """
        Update the next item that needs clarification with the provided details.
        Sets status to CONFIRMED and updates price/customizations.
        """
        next_clar = self.get_next_clarification()
        if next_clar:
            idx, item = next_clar
            if item.status != ItemStatus.CONFIRMED:
                item.price = price
                item.customizations = customizations or {}
                item.status = ItemStatus.CONFIRMED
                item.clarification_needed = None
                self.calculate_total()
                return True
        return False

    def update_item_customizations(self, item_id: str, customizations: Dict[str, str], price: float = None):
        # VALIDATE: Ensure item exists in menu
        menu_item = processor.get_menu_item(item_id)
        if not menu_item:
            raise ValueError(f"Item with id '{item_id}' does not exist in the menu.")
        # VALIDATE: Ensure price matches menu if provided
        if price is not None and float(price) != float(menu_item.price):
            raise ValueError(f"Price mismatch for item id '{item_id}': got '{price}', expected '{menu_item.price}'")
        for item in self.items:
            if item.id == item_id:
                item.customizations = customizations
                if price is not None:
                    # Adjust total if price changes
                    item.price = price
                    if item.status == ItemStatus.CONFIRMED:
                        self.calculate_total()
                    
                item.status = ItemStatus.CONFIRMED
                item.clarification_needed = None
                
                return True
        return False

    def calculate_total(self):
        """Calculate the total price of all confirmed items"""
        self.total = sum(item.price * item.quantity for item in self.items if item.status == ItemStatus.CONFIRMED)

    def to_dict(self):
        return {
            "items": [item.to_dict() for item in self.items],
            "total": round(self.total, 2),
            "current_clarification_index": self.current_clarification_index
        }

app = Flask(__name__)
load_dotenv()

# Initialize OpenAI client
azure_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-05-15"
)

# Initialize Gemini client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-pro')

# Initialize menu processor
menu_file = os.path.join(os.path.dirname(__file__), 'mcdonalds_fixed_full.json')
menu_items = processor.process_menu(menu_file)
print(f"Processed {menu_items} menu items")

# Print loaded menu items for debugging
for item_id, item in processor.menu_items.items():
    print(f"Loaded item: {item_id} - {item.name}")

# Track order state
current_orders: Dict[str, Order] = {}
# Track LangChain memory per session
session_memories = {}

def get_memory(session_id):
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(return_messages=True)
    return session_memories[session_id]

def generate_response(message, session_id, model="gpt-4"):
    try:
        # Get or create order for this session
        if session_id not in current_orders:
            current_orders[session_id] = Order()
        current_order = current_orders[session_id]
        
        # Get conversation memory for this session
        memory = get_memory(session_id)
        
        # Log the incoming message
        chat_logger.info(f"User message: {message}")
        
        # Get completion based on selected model
        if model == "gemini-2.5":
            # Use Gemini model
            chat = gemini_model.start_chat(history=[])
            response = chat.send_message(message)
            ai_response = response.text
        else:
            # Use Azure OpenAI
            response = azure_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *memory.chat_memory.messages,
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=800
            )
            ai_response = response.choices[0].message.function_call.arguments if response.choices[0].message.function_call else response.choices[0].message.content
        
        print(f"AI Response: {ai_response}")
        
        try:
            response_data = json.loads(ai_response)
        except json.JSONDecodeError:
            return ai_response
            
        print(f"Parsed JSON: {json.dumps(response_data, indent=2)}")
        
        if isinstance(response_data, dict) and response_data.get("type") == "order_update":
            # Backend guardrail: correct any LLM mistakes about non-required clarifications
            for item_data in response_data.get("items", []):
                item_id = item_data.get("id")
                clarification_needed = item_data.get("clarification_needed")
                menu_item = processor.get_menu_item(item_id)
                # If clarification_needed is set for a non-required customization, override to confirmed
                if (
                    clarification_needed
                    and menu_item
                    and hasattr(menu_item, "customization_options")
                    and clarification_needed in menu_item.customization_options
                ):
                    cust_option = menu_item.customization_options[clarification_needed]
                    # cust_option can be a dict or list; handle both
                    if isinstance(cust_option, dict):
                        required = cust_option.get("required", False)
                    elif isinstance(cust_option, list):
                        # If list, not required
                        required = False
                    else:
                        required = False
                    if not required:
                        item_data["status"] = "confirmed"
                        item_data["clarification_needed"] = None
            # Process order updates
            for item_data in response_data.get("items", []):
                item_id = item_data.get("id")
                price = float(item_data.get("price", 0))
                customizations = item_data.get("customizations", {})
                status = item_data.get("status", "confirmed")
                clarification_needed = item_data.get("clarification_needed")

                # Check if item is already in order; if so, update it instead of adding a new one
                updated = current_order.update_item_customizations(
                    item_id=item_id,
                    customizations=customizations,
                    price=price
                )
                if not updated:
                    # Validate that the item exists in the menu
                    menu_item = processor.get_menu_item(item_id)
                    if not menu_item:
                        chat_logger.error(f"Item with id '{item_id}' does not exist in the menu")
                        return f"I apologize, but I couldn't find the item '{item_data.get('name', 'unknown')}' in our menu. Could you please try ordering something else?"
                    
                    current_order.add_item(
                        item_id=item_id,
                        name=item_data.get('name', menu_item.name),  # Use menu item name if not provided
                        price=price,
                        status=ItemStatus(status),
                        clarification_needed=clarification_needed,
                        customizations=customizations
                    )
            # Add AI response to conversation memory
            memory.chat_memory.add_ai_message(response_data.get("message", "I've updated your order. What else can I help you with?"))
            # Get a human-readable response
            return response_data.get("message", "I've updated your order. What else can I help you with?")
        # If we have a pending clarification, try to process the response
        # Advanced: If user message mentions a specific pending item's name, update that item
        pending_items = [(i, item) for i, item in enumerate(current_order.items) if item.status != ItemStatus.CONFIRMED]
        # Try to find which pending item the user is referring to
        matched_index = None
        user_msg_lower = message.lower()
        for idx, pending_item in pending_items:
            if pending_item.name.lower() in user_msg_lower:
                matched_index = idx
                break
        if matched_index is None and pending_items:
            # Default to the first pending item
            matched_index, item = pending_items[0]
        else:
            item = current_order.items[matched_index] if matched_index is not None else None
        if item:
            menu_item = processor.get_menu_item(item.id)
            if menu_item:
                if item.status == ItemStatus.NEEDS_CUSTOMIZATION and menu_item.customization_options:
                    customizations = item.customizations.copy() if item.customizations else {}
                    for option_type, options_data in menu_item.customization_options.items():
                        available_options = options_data.get('options', [])
                        for opt in available_options:
                            opt_name = opt['name'].lower() if isinstance(opt, dict) and 'name' in opt else str(opt).lower()
                            if opt_name in user_msg_lower:
                                customizations[option_type] = opt['name'] if isinstance(opt, dict) and 'name' in opt else opt
                    is_valid, _ = processor.validate_customization(item.id, customizations)
                    if is_valid:
                        current_order.update_next_clarification(price=item.price, customizations=customizations)
                        return f"I've updated your {item.name} with your customizations: {customizations}."
                    else:
                        return "I couldn't match your customization to available options. Please specify valid options."
        # If the user's message refers to a confirmed item and contains valid customizations, allow updating
        # Search all items (confirmed and pending) for a name match
        for idx, order_item in enumerate(current_order.items):
            if order_item.name.lower() in user_msg_lower and order_item.status == ItemStatus.CONFIRMED:
                menu_item = processor.get_menu_item(order_item.id)
                if menu_item and menu_item.customization_options:
                    customizations = order_item.customizations.copy() if order_item.customizations else {}
                    for option_type, options_data in menu_item.customization_options.items():
                        available_options = options_data.get('options', [])
                        for opt in available_options:
                            opt_name = opt['name'].lower() if isinstance(opt, dict) and 'name' in opt else str(opt).lower()
                            if opt_name in user_msg_lower:
                                customizations[option_type] = opt['name'] if isinstance(opt, dict) and 'name' in opt else opt
                    is_valid, _ = processor.validate_customization(order_item.id, customizations)
                    if is_valid:
                        # Update the confirmed item with new customizations
                        order_item.customizations = customizations
                        return f"I've updated your {order_item.name} with your new customizations: {customizations}."
                    else:
                        return f"I couldn't match your customization to available options for {order_item.name}. Please specify valid options."
        print(f"[DEBUG] Order after processing response: {current_order.to_dict()}")
        return ai_response

    except Exception as e:
        error_msg = f"An error occurred while processing your request: {str(e)}"
        chat_logger.error(error_msg)
        return error_msg

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/send_message", methods=["POST"])
def send_message():
    data = request.get_json()
    session_id = data.get('session_id')
    message = data.get('message')
    model = data.get('model', 'gpt-4')  # Default to GPT-4 if not specified
    
    try:
        # Log user message
        chat_logger.info(f"USER ({session_id}): {message}")
        response = generate_response(message, session_id, model)
        # Log bot response
        chat_logger.info(f"BOT ({session_id}): {response}")
        print(f"[DEBUG] Current order after response: {current_orders[session_id].to_dict()}")
        # Include current order in response
        order_data = current_orders[session_id].to_dict() if session_id in current_orders else {"items": [], "total": 0}
        return jsonify({
            'response': response,
            'order': order_data
        })
    except Exception as e:
        error_message = f"Sorry, there was an error processing your request: {str(e)}"
        # Log the error response
        chat_logger.error(f"ERROR ({session_id}): {error_message}")
        return jsonify({'error': error_message})

@app.route('/update_order', methods=['POST'])
def update_order():
    data = request.json
    session_id = data.get('session_id')
    action = data.get('action')
    
    if not session_id or session_id not in current_orders:
        current_orders[session_id] = Order()
    
    order = current_orders[session_id]
    
    if action == 'add':
        items = data.get('items', [])
        for item_data in items:
            # Convert status string to enum
            status_str = item_data.get('status', 'pending')
            try:
                status = ItemStatus(status_str)
            except ValueError:
                status = ItemStatus.PENDING
                
            # Validate that the item exists in the menu
            item_id = item_data.get('id')
            if not processor.get_menu_item(item_id):
                return jsonify({
                    'error': f"Item with id '{item_id}' does not exist in the menu"
                }), 400
                
            order.add_item(
                item_id=item_id,
                name=item_data.get('name', ''),
                price=float(item_data.get('price', 0)),
                status=status,
                clarification_needed=item_data.get('clarification_needed')
            )
    elif action == 'remove':
        index = data.get('index')
        if index is not None and 0 <= index < len(order.items):
            order.remove_item(index)
    
    # Return the complete order data
    return jsonify({
        'items': [item.to_dict() for item in order.items],
        'total': order.total
    })

@app.route("/menu_item/<item_id>")
def get_menu_item(item_id):
    menu_item = processor.get_menu_item(item_id)
    if not menu_item:
        return jsonify({"error": "Item not found"}), 404
    return jsonify({
        "id": menu_item.id,
        "name": menu_item.name,
        "price": menu_item.price,
        "description": menu_item.description,
        "category": menu_item.category,
        "customization_options": menu_item.customization_options
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)
