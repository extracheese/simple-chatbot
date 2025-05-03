import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import datetime

# Set up logging (can be configured globally later)
chat_logger = logging.getLogger('chat')

class OrderStatus(Enum):
    """Represents the status of an order."""
    PENDING = "Pending"
    CONFIRMED = "Confirmed"
    CANCELLED = "Cancelled"
    COMPLETED = "Completed"

@dataclass
class Order:
    """Represents a customer's order."""
    order_id: str
    session_id: str
    items: List[Dict] = field(default_factory=list)
    status: OrderStatus = OrderStatus.PENDING
    total_amount: float = 0.0
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    notes: Optional[str] = None

    def __post_init__(self):
        # Ensure status is an OrderStatus enum
        if isinstance(self.status, str):
            try:
                self.status = OrderStatus(self.status)
            except ValueError:
                chat_logger.warning(f"Invalid status string '{self.status}' for order {self.order_id}. Defaulting to PENDING.")
                self.status = OrderStatus.PENDING
        elif not isinstance(self.status, OrderStatus):
            chat_logger.warning(f"Invalid status type '{type(self.status)}' for order {self.order_id}. Defaulting to PENDING.")
            self.status = OrderStatus.PENDING

        self.calculate_total()

    def add_item(self, item_id: str, name: str, quantity: int, price: float, notes: Optional[str] = None):
        """Adds an item to the order or updates its quantity if it already exists."""
        for existing_item in self.items:
            if existing_item['id'] == item_id:
                chat_logger.info(f"Updating quantity for existing item {item_id} in order {self.order_id}.")
                existing_item['quantity'] += quantity
                if notes:
                    existing_item['notes'] = notes # Overwrite or append notes as needed
                self.calculate_total()
                self.touch()
                return

        # Item not found, add new one
        new_item = {
            'id': item_id,
            'name': name,
            'quantity': quantity,
            'price': price, # Price per unit
            'notes': notes
        }
        self.items.append(new_item)
        chat_logger.info(f"Added new item {item_id} to order {self.order_id}.")
        self.calculate_total()
        self.touch()

    def modify_item(self, item_id: str, new_quantity: int, new_notes: Optional[str] = None) -> bool:
        """Modifies an existing item's quantity or notes. Removes if quantity is zero or less."""
        item_found = False
        item_to_remove = None
        for item in self.items:
            if item['id'] == item_id:
                if new_quantity <= 0:
                    chat_logger.info(f"Quantity for item {item_id} set to {new_quantity}. Marking for removal from order {self.order_id}.")
                    item_to_remove = item
                else:
                    chat_logger.info(f"Modifying item {item_id} in order {self.order_id}. New quantity: {new_quantity}.")
                    item['quantity'] = new_quantity
                    if new_notes is not None: # Allow setting notes to empty string
                        item['notes'] = new_notes
                item_found = True
                break
        
        if item_to_remove:
            self.items.remove(item_to_remove)
            chat_logger.info(f"Removed item {item_id} from order {self.order_id}.")

        if item_found:
            self.calculate_total()
            self.touch()
            return True
        else:
            chat_logger.warning(f"Attempted to modify non-existent item {item_id} in order {self.order_id}.")
            return False

    def calculate_total(self):
        """Recalculates the total amount for the order."""
        self.total_amount = sum(item['quantity'] * item['price'] for item in self.items)
        chat_logger.debug(f"Recalculated total for order {self.order_id}: ${self.total_amount:.2f}")

    def set_status(self, status: OrderStatus):
        """Sets the order status."""
        if isinstance(status, OrderStatus):
            self.status = status
            chat_logger.info(f"Set status for order {self.order_id} to {status.value}.")
            self.touch()
        else:
             chat_logger.warning(f"Attempted to set invalid status type '{type(status)}' for order {self.order_id}.")

    def touch(self):
        """Updates the updated_at timestamp."""
        self.updated_at = datetime.datetime.now()

    def to_dict(self) -> Dict:
        """Converts the Order object to a dictionary suitable for JSON serialization."""
        return {
            'order_id': self.order_id,
            'session_id': self.session_id,
            'items': self.items,
            'status': self.status.value, # Use enum value for serialization
            'total_amount': self.total_amount,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'notes': self.notes
        }
