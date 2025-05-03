import logging
import uuid
from typing import Dict, Optional, List, Any, Tuple
from langchain.memory.buffer import ConversationBufferMemory
from .order import Order, OrderStatus
from menu.processor import MenuProcessor, MenuItem

# Get logger instance (ensure logging is configured elsewhere, e.g., at app start)
chat_logger = logging.getLogger('chat')

# In-memory storage (replace with persistent storage for production)
_current_orders: Dict[str, Order] = {}
_session_memories: Dict[str, ConversationBufferMemory] = {}

def get_memory(session_id: str) -> ConversationBufferMemory | None:
    """Retrieves the memory buffer for a given session ID."""
    return _session_memories.get(session_id)

def update_memory(session_id: str, memory: ConversationBufferMemory):
    """Updates or adds the memory buffer for a given session ID."""
    _session_memories[session_id] = memory
    chat_logger.debug(f"Session {session_id} - Memory updated/created in state manager.")

def get_or_create_memory(session_id: str) -> ConversationBufferMemory:
    """Gets the existing memory or creates a new one if none exists."""
    if session_id not in _session_memories:
        chat_logger.info(f"Session {session_id} - Initialized new memory buffer.")
        _session_memories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return _session_memories[session_id]

def get_order(session_id: str) -> Order | None:
    """Retrieves the order for a given session ID."""
    return _current_orders.get(session_id)

def update_order(session_id: str, order: Order):
    """Updates or adds the order for a given session ID."""
    _current_orders[session_id] = order
    chat_logger.debug(f"Session {session_id} - Order {order.order_id} updated/created in state manager.")

def get_or_create_order(session_id: str) -> Order:
    """Gets the existing order or creates a new one if none exists."""
    if session_id not in _current_orders:
        new_order_id = str(uuid.uuid4())
        chat_logger.info(f"Session {session_id} - Initialized new order object: {new_order_id}")
        _current_orders[session_id] = Order(order_id=new_order_id, session_id=session_id)
    return _current_orders[session_id]

def load_order_from_dict(session_id: str, current_order_dict: Dict[str, Any], processor: MenuProcessor) -> Order:
    """Loads or updates an order from a dictionary, validating items against the menu.
    
    Args:
        session_id: The ID of the session.
        current_order_dict: The dictionary containing order data to load.
        processor: The MenuProcessor instance for item validation.
        
    Returns:
        The loaded or updated Order object.
        
    Raises:
        ValueError: If an item in the dictionary cannot be found in the menu.
    """
    order_id = current_order_dict.get('order_id', str(uuid.uuid4()))
    
    if session_id in _current_orders and _current_orders[session_id].order_id == order_id:
        # Update existing order instance
        order = _current_orders[session_id]
        chat_logger.info(f"Session {session_id} - Updating existing order {order_id} from dict.")
        # Update status carefully, handling potential string value
        status_val = current_order_dict.get('status', OrderStatus.PENDING.value) 
        try:
            order.status = OrderStatus(status_val)
        except ValueError:
            order.status = OrderStatus.PENDING
            chat_logger.warning(f"Session {session_id} - Invalid status '{status_val}' loaded for order {order_id}. Defaulting to PENDING.")
        
        # Validate and load items
        loaded_items_raw = current_order_dict.get('items', [])
        validated_items: List[Any] = []
        for item_data in loaded_items_raw:
            item_id = item_data.get('item_id') or item_data.get('name') # Allow 'name' for compatibility
            quantity = item_data.get('quantity', 1)
            
            if not item_id:
                chat_logger.warning(f"Session {session_id} - Skipping item load: missing 'item_id' or 'name' in {item_data}")
                continue
                
            # Validate against menu
            menu_item = processor.get_menu_item(item_id)
            if not menu_item:
                chat_logger.error(f"Session {session_id} - Item '{item_id}' in update request not found in menu.")
                raise ValueError(f"Item '{item_id}' not found on the menu.")
            
            # Use validated data
            validated_items.append({
                'item_id': menu_item['id'], # Use canonical ID
                'name': menu_item['name'],  # Use canonical name
                'quantity': quantity,
                'price': menu_item.get('price', 0.0)
            })
            
        order.items = validated_items # Assign validated list
        order.notes = current_order_dict.get('notes')
        # Assuming created_at shouldn't change, update updated_at
        order.touch() 
        order.calculate_total() # Recalculate total after loading items

    else:
        # Create new order instance if not found or ID mismatch
        chat_logger.info(f"Session {session_id} - Creating new order instance {order_id} from dict.")
        
        # Validate and load items for new order too
        loaded_items_raw = current_order_dict.get('items', [])
        validated_items = []
        for item_data in loaded_items_raw:
            item_id = item_data.get('item_id') or item_data.get('name')
            quantity = item_data.get('quantity', 1)
            if not item_id:
                chat_logger.warning(f"Session {session_id} - Skipping item load during creation: missing 'item_id' or 'name' in {item_data}")
                continue
            menu_item = processor.get_menu_item(item_id)
            if not menu_item:
                chat_logger.error(f"Session {session_id} - Item '{item_id}' in creation request not found in menu.")
                raise ValueError(f"Item '{item_id}' not found on the menu.")
            validated_items.append({
                'item_id': menu_item['id'], 'name': menu_item['name'], 
                'quantity': quantity, 'price': menu_item.get('price', 0.0)
            })
            
        order = Order(
            order_id=order_id,
            session_id=session_id,
            items=validated_items, # Use validated list
            status=current_order_dict.get('status', OrderStatus.PENDING.value), # Handled by Order.__post_init__
            notes=current_order_dict.get('notes')
            # Timestamps are handled by default factory or Order.__post_init__
        )
        _current_orders[session_id] = order # Store the new instance

    return order
