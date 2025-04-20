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
    item_id: str
    name: str
    status: ItemStatus = ItemStatus.CONFIRMED
    clarification_needed: Optional[str] = None
    customizations: Dict[str, str] = field(default_factory=dict)
    quantity: int = 1

    def get_price(self) -> float:
        """Get the current price for this item based on menu and customizations"""
        menu_item = processor.get_menu_item(self.item_id)
        if not menu_item:
            return 0.0
        size = self.customizations.get('size')
        return menu_item.get_price_for_size(size)

class Order:
    def __init__(self):
        self.items: List[OrderItem] = []

    def add_item(self, item_id: str, name: str = None, status: ItemStatus = ItemStatus.CONFIRMED, 
                 clarification_needed: str = None, customizations: Dict[str, str] = None):
        """Add an item to the order"""
        menu_item = processor.get_menu_item(item_id)
        if not menu_item:
            raise ValueError(f"Item with id '{item_id}' does not exist")

        # Create order item
        order_item = OrderItem(
            item_id=item_id,
            name=name or menu_item.name,
            status=status,
            clarification_needed=clarification_needed,
            customizations=customizations or {}
        )
        self.items.append(order_item)
        return order_item

    def remove_item(self, index: int):
        if 0 <= index < len(self.items):
            self.items.pop(index)

    def update_item_customizations(self, item_id: str, customizations: Dict[str, str], index: Optional[int] = None):
        """Update customizations for an item"""
        menu_item = processor.get_menu_item(item_id)
        if not menu_item:
            raise ValueError(f"Item with id '{item_id}' does not exist")

        # Find the item to update
        if index is not None and 0 <= index < len(self.items):
            item = self.items[index]
            if item.item_id != item_id:
                raise ValueError(f"Item at index {index} has id '{item.item_id}', expected '{item_id}'")
        else:
            # Find the first item with matching id
            item = next((item for item in self.items if item.item_id == item_id), None)
            if not item:
                return False

        # Update customizations
        item.customizations.update(customizations)
        return True

    def get_total(self) -> float:
        """Calculate the total price of the order based on menu prices"""
        total = 0.0
        for item in self.items:
            if item.status == ItemStatus.CONFIRMED:
                total += item.get_price() * item.quantity
        return round(total, 2)

    def get_items(self) -> List[Dict]:
        """Get all items in the order with their current prices"""
        items = []
        for item in self.items:
            # Convert status to string if it's an enum
            status = item.status.value if isinstance(item.status, ItemStatus) else item.status
            items.append({
                'id': item.item_id,
                'name': item.name,
                'status': status,
                'clarification_needed': item.clarification_needed,
                'customizations': item.customizations,
                'quantity': item.quantity,
                'price': item.get_price()
            })
        return items

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
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            # Add chat memory messages
            if memory and memory.chat_memory and memory.chat_memory.messages:
                for msg in memory.chat_memory.messages:
                    messages.append({
                        "role": "user" if msg.type == "human" else "assistant",
                        "content": msg.content
                    })
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Get completion
            response = azure_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            ai_response = response.choices[0].message.function_call.arguments if response.choices[0].message.function_call else response.choices[0].message.content
            
            # Debug logging
            chat_logger.debug(f"Raw AI response: {ai_response}")
            
            try:
                response_data = json.loads(ai_response)
                chat_logger.debug(f"Parsed response data: {json.dumps(response_data, indent=2)}")
                
                # Validate response against schema
                if not isinstance(response_data, dict):
                    chat_logger.error(f"Response is not a dictionary: {response_data}")
                    return "I apologize, but I couldn't process your order correctly. Could you please try again?"
                
                if response_data.get("type") != "order_update":
                    chat_logger.error(f"Response type is not order_update: {response_data.get('type')}")
                    return response_data.get("message", ai_response)
                
                if "items" not in response_data:
                    chat_logger.error(f"No items in response: {response_data}")
                    return "I apologize, but I couldn't process your order correctly. Could you please try again?"
                
                # Validate each item
                for item in response_data["items"]:
                    if "id" not in item:
                        chat_logger.error(f"Item missing id: {item}")
                        return "I apologize, but I couldn't process your order correctly. Could you please try again?"
                    if not processor.get_menu_item(item["id"]):
                        chat_logger.error(f"Invalid item id: {item['id']}")
                        return f"I apologize, but I couldn't find the item '{item.get('name', 'unknown')}' in our menu. Could you please try ordering something else?"
                
                if isinstance(response_data, dict) and response_data.get("type") == "order_update":
                    # Backend guardrail: correct any LLM mistakes about non-required clarifications
                    for item_data in response_data.get("items", []):
                        item_id = item_data.get("id")
                        clarification_needed = item_data.get("clarification_needed")
                        menu_item = processor.get_menu_item(item_id)
                        
                        # Validate size if provided
                        customizations = item_data.get("customizations", {})
                        if "size" in customizations:
                            size = customizations["size"]
                            if (
                                not hasattr(menu_item, "customization_options")
                                or "size" not in menu_item.customization_options
                                or size not in menu_item.customization_options["size"]["options"]
                            ):
                                chat_logger.error(f"Invalid size '{size}' for item '{item_id}'")
                                return f"I apologize, but '{size}' is not a valid size for that item. Please try again with a valid size option."
                        
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
                        customizations = item_data.get("customizations", {})
                        status = ItemStatus(item_data.get("status", "confirmed"))  # Convert string to enum
                        clarification_needed = item_data.get("clarification_needed")

                        # Check if item is already in order; if so, update it instead of adding a new one
                        updated = current_order.update_item_customizations(
                            item_id=item_id,
                            customizations=customizations
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
                                status=status,  # Pass the enum directly
                                clarification_needed=clarification_needed,
                                customizations=customizations
                            )
                    # Add AI response to conversation memory
                    memory.chat_memory.add_user_message(message)
                    memory.chat_memory.add_ai_message(response_data.get("message", "I've updated your order. What else can I help you with?"))
                    # Get a human-readable response
                    return response_data.get("message", "I've updated your order. What else can I help you with?")
            except json.JSONDecodeError:
                chat_logger.error(f"Failed to parse AI response as JSON: {ai_response}")
                return ai_response
            except Exception as e:
                chat_logger.error(f"Error processing AI response: {str(e)}")
                return "I apologize, but I couldn't process your order correctly. Could you please try again?"
            
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
                menu_item = processor.get_menu_item(item.item_id)
                if menu_item:
                    if item.status == ItemStatus.NEEDS_CUSTOMIZATION and menu_item.customization_options:
                        customizations = item.customizations.copy() if item.customizations else {}
                        for option_type, options_data in menu_item.customization_options.items():
                            available_options = options_data.get('options', [])
                            for opt in available_options:
                                opt_name = opt['name'].lower() if isinstance(opt, dict) and 'name' in opt else str(opt).lower()
                                if opt_name in user_msg_lower:
                                    customizations[option_type] = opt['name'] if isinstance(opt, dict) and 'name' in opt else opt
                        is_valid, _ = processor.validate_customization(item.item_id, customizations)
                        if is_valid:
                            current_order.update_item_customizations(item.item_id, customizations)
                            return f"I've updated your {item.name} with your customizations: {customizations}."
                        else:
                            return "I couldn't match your customization to available options. Please specify valid options."
            # If the user's message refers to a confirmed item and contains valid customizations, allow updating
            # Search all items (confirmed and pending) for a name match
            for idx, order_item in enumerate(current_order.items):
                if order_item.name.lower() in user_msg_lower and order_item.status == ItemStatus.CONFIRMED:
                    menu_item = processor.get_menu_item(order_item.item_id)
                    if menu_item and menu_item.customization_options:
                        customizations = order_item.customizations.copy() if order_item.customizations else {}
                        for option_type, options_data in menu_item.customization_options.items():
                            available_options = options_data.get('options', [])
                            for opt in available_options:
                                opt_name = opt['name'].lower() if isinstance(opt, dict) and 'name' in opt else str(opt).lower()
                                if opt_name in user_msg_lower:
                                    customizations[option_type] = opt['name'] if isinstance(opt, dict) and 'name' in opt else opt
                        is_valid, _ = processor.validate_customization(order_item.item_id, customizations)
                        if is_valid:
                            # Update the confirmed item with new customizations
                            order_item.customizations = customizations
                            return f"I've updated your {order_item.name} with your new customizations: {customizations}."
                        else:
                            return f"I couldn't match your customization to available options for {order_item.name}. Please specify valid options."
            print(f"[DEBUG] Order after processing response: {current_order.get_items()}")
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
    try:
        data = request.get_json()
        message = data.get("message")
        session_id = data.get("session_id")
        model = data.get("model", "gpt-4")

        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400

        # Get or create order for this session
        if session_id not in current_orders:
            current_orders[session_id] = Order()
        current_order = current_orders[session_id]

        # Get chat memory for this session
        if session_id not in session_memories:
            session_memories[session_id] = ConversationBufferMemory(return_messages=True)
        memory = session_memories[session_id]

        # Generate response
        response = generate_response(message, session_id, model)

        # Return response with current order data
        return jsonify({
            "response": response,
            "order": {
                "items": current_order.get_items(),
                "total": current_order.get_total()
            }
        })

    except Exception as e:
        error_msg = f"An error occurred while processing your request: {str(e)}"
        chat_logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route("/update_order", methods=["POST"])
def update_order():
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400
        
        if session_id not in current_orders:
            current_orders[session_id] = Order()
        current_order = current_orders[session_id]
        
        # Process order updates
        for item_data in data.get("items", []):
            item_id = item_data.get("id")
            status = item_data.get("status", "confirmed")
            clarification_needed = item_data.get("clarification_needed")
            customizations = item_data.get("customizations", {})
            
            # Validate menu item exists
            menu_item = processor.get_menu_item(item_id)
            if not menu_item:
                return jsonify({"error": f"Item with id '{item_id}' does not exist"}), 400
            
            # Validate size if provided
            if "size" in customizations:
                size = customizations["size"]
                if (
                    not hasattr(menu_item, "customization_options")
                    or "size" not in menu_item.customization_options
                    or size not in menu_item.customization_options["size"]["options"]
                ):
                    return jsonify({"error": f"Invalid size '{size}' for item '{item_id}'"}), 400
            
            # Check if item exists in order
            updated = current_order.update_item_customizations(
                item_id=item_id,
                customizations=customizations
            )
            if not updated:
                # Add new item
                current_order.add_item(
                    item_id=item_id,
                    name=item_data.get('name', menu_item.name),
                    status=status,
                    clarification_needed=clarification_needed,
                    customizations=customizations
                )
        
        return jsonify({
            "message": "Order updated successfully",
            "order": {
                "items": current_order.get_items(),
                "total": current_order.get_total()
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
