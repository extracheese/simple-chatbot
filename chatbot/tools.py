import logging
import json
from typing import Tuple, Dict, Any, Optional

# Import necessary components from other modules
from .order import Order, OrderStatus # Assuming Order, OrderStatus are in order.py
from menu.processor import MenuProcessor # Assuming MenuProcessor is in menu/processor.py

# Get logger instance (ensure logging is configured before use)
chat_logger = logging.getLogger('chat')
menu_logger = logging.getLogger('menu.processor') # Match logger name used in MenuProcessor

# --- Tool/Function Definitions --- #

def search_menu(processor: MenuProcessor, query: str, current_order: Optional[Order] = None) -> Tuple[str, bool]:
    """Searches the menu for items matching the query. 
    If an exact match is found, returns details of that item.
    Otherwise, returns items containing the query term.
    Also returns a boolean indicating if an exact match was found.
    Args:
        processor: The MenuProcessor instance.
        query: The search term from the user.
        current_order: The current order object (unused in this specific function but kept for potential future use or consistency).

    Returns:
        A tuple containing:
         - A string describing the found menu items or a confirmation.
         - A boolean: True if an exact match was found, False otherwise.
    """
    chat_logger.info(f"Tool: search_menu called with query: '{query}'")
    
    # Use the enhanced find_menu_items which returns (results_string, found_exact)
    result_string, found_exact = processor.find_menu_items(query) 
    
    chat_logger.info(f"Tool: search_menu results: Exact match found = {found_exact}, Result string length: {len(result_string)}")
    # Limit the length of the result string logged to avoid excessive noise
    log_preview = result_string[:200] + ('...' if len(result_string) > 200 else '')
    chat_logger.debug(f"Tool: search_menu result string preview: {log_preview}")

    return result_string, found_exact

def update_order_tool(session_id: str, order_updates: Dict[str, Any], current_order: Order, processor: MenuProcessor) -> str:
    """Adds, modifies, or removes items from the current order based on provided updates.
       Validates items against the menu.
    Args:
        session_id: The identifier for the user's session.
        order_updates: A dictionary containing 'action' ('add', 'remove', 'update') and 'items' list.
                       Each item in the list should have 'name' and optionally 'quantity'.
        current_order: The current Order object for the session.
        processor: The MenuProcessor instance for validation.

    Returns:
        A string confirming the order update or detailing issues.
    """
    chat_logger.info(f"Session {session_id} - Tool: update_order_tool called with updates: {order_updates}") 
    action = order_updates.get('action', 'add').lower() # Default to 'add'
    items = order_updates.get('items', [])
    
    if not items:
        return "No items provided to update in the order."

    update_messages = []
    items_processed_count = 0
    items_failed_count = 0

    for item_data in items:
        item_name = item_data.get('name')
        quantity_str = item_data.get('quantity', '1') # Default quantity to 1 if not specified

        if not item_name:
            chat_logger.warning(f"Session {session_id} - Skipping item update: missing 'name' in {item_data}")
            update_messages.append(f"Skipped an item because its name was missing.")
            items_failed_count += 1
            continue
            
        # Validate quantity
        try:
            quantity = int(quantity_str)
            if quantity <= 0:
                 raise ValueError("Quantity must be positive")
        except ValueError:
             chat_logger.warning(f"Session {session_id} - Invalid quantity '{quantity_str}' for item '{item_name}'. Defaulting to 1.")
             update_messages.append(f"Invalid quantity '{quantity_str}' for '{item_name}', assuming 1.")
             quantity = 1

        # Validate item exists using MenuProcessor
        chat_logger.debug(f"Session {session_id} - Attempting menu lookup for item name: {repr(item_name)}")
        menu_item = processor.get_menu_item(item_name)
        if not menu_item:
            chat_logger.warning(f"Session {session_id} - Item '{item_name}' not found in menu.")
            update_messages.append(f"Could not find '{item_name}' on the menu.")
            items_failed_count += 1
            continue # Skip adding/modifying if item doesn't exist
        
        # Ensure we use the canonical name and price from the menu
        # MenuItem is a class with attributes, not a dictionary
        canonical_name = menu_item.name  # Use attribute access instead of dictionary access
        price = menu_item.price  # Use attribute access instead of dictionary access

        if action == 'add':
            current_order.add_item(item_id=canonical_name, name=canonical_name, quantity=quantity, price=price)
            update_messages.append(f"Added {quantity} x {canonical_name}.")
            items_processed_count += 1
        elif action == 'update': # Update implies changing quantity or details
            success = current_order.update_item_quantity(item_id=canonical_name, new_quantity=quantity)
            if success:
                update_messages.append(f"Updated {canonical_name} quantity to {quantity}.")
                items_processed_count += 1
            else:
                # If update fails, maybe add it instead?
                current_order.add_item(item_id=canonical_name, name=canonical_name, quantity=quantity, price=price)
                update_messages.append(f"Added {quantity} x {canonical_name} (was not in order previously).")
                items_processed_count += 1
        elif action == 'remove':
            success = current_order.remove_item(item_id=canonical_name)
            if success:
                update_messages.append(f"Removed {canonical_name} from the order.")
                items_processed_count += 1
            else:
                 chat_logger.warning(f"Session {session_id} - Tried to remove item '{canonical_name}' which was not in the order.")
                 update_messages.append(f"'{canonical_name}' wasn't in your order to remove.")
                 # Consider this a 'soft' failure - the item wasn't removed, but it wasn't there anyway
                 # items_failed_count += 1 
        else:
            chat_logger.warning(f"Session {session_id} - Invalid action '{action}' received.")
            update_messages.append(f"Unknown action '{action}' for item '{canonical_name}'.")
            items_failed_count += 1

    # Construct summary message
    summary = "Order updated: " + " ".join(update_messages)
    if items_failed_count > 0:
        summary += f" ({items_failed_count} item(s) could not be processed fully)." 
        
    # Add current order details if anything was processed
    if items_processed_count > 0:
        current_total = current_order.calculate_total()
        # Check if current_total is None or a valid number
        if current_total is not None:
            summary += f" Your current total is ${current_total:.2f}."
        else:
            summary += f" Your current total is $0.00."  # Default to 0 if None
        chat_logger.info(f"Session {session_id} - Order after update: {current_order.to_dict()}")
    else:
        chat_logger.warning(f"Session {session_id} - No items were successfully processed in update_order_tool call.")
        if not update_messages: # Handle case where loop didn't run (e.g., empty items list initially)
            summary = "No changes were made to the order based on the request."

    return summary

def confirm_order(session_id: str, current_order: Order) -> str:
    """Confirms the current order, calculates the total, and sets the status to CONFIRMED.
    Args:
        session_id: The identifier for the user's session.
        current_order: The current Order object.

    Returns:
        A string confirming the order and stating the total price.
    """
    chat_logger.info(f"Session {session_id} - Tool: confirm_order called.")
    if not current_order.items:
        chat_logger.warning(f"Session {session_id} - Attempted to confirm an empty order.")
        return "Your order is currently empty. Please add items before confirming."
        
    current_order.status = OrderStatus.CONFIRMED
    total_price = current_order.calculate_total()
    confirmation_message = f"OK. Your order is confirmed with {len(current_order.items)} item(s). Your total is ${total_price:.2f}. Thank you!"
    chat_logger.info(f"Session {session_id} - Order confirmed. Final state: {current_order.to_dict()}")
    return confirmation_message

# --- Tool Schema Definitions (for LLM) --- #

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "search_menu",
            "description": "Searches the restaurant menu for items based on a query. Use this to find items, check availability, or get details like price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The food item or category to search for (e.g., 'Big Mac', 'fries', 'happy meal', 'drinks')."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_order_tool",
            "description": "Adds, removes, or updates the quantity of items in the current order. Use 'add' to add new items, 'remove' to delete items, and 'update' to change the quantity of existing items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "remove", "update"],
                        "description": "The action to perform: 'add' items, 'remove' items, or 'update' item quantities."
                    },
                    "items": {
                        "type": "array",
                        "description": "A list of items to modify in the order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The name of the menu item (e.g., 'McChicken', 'Large Coke')."
                                },
                                "quantity": {
                                    "type": "string", # Keep as string to handle variations LLM might send
                                    "description": "The number of this item (e.g., '2', '1'). Required for 'add' and 'update'."
                                }
                            },
                            "required": ["name"]
                        }
                    }
                },
                "required": ["action", "items"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_order",
            "description": "Confirms the user's current order, calculates the final total, and marks the order as complete. Use only when the user explicitly agrees to finalize the order.",
            "parameters": {
                "type": "object",
                "properties": {},
                 # No specific parameters needed other than the implicit current_order
                 "required": []
            }
        }
    }
]
