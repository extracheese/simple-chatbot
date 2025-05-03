from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import datetime
import logging

# Assuming chat_logger is configured elsewhere (e.g., application.py)
# If not, basicConfig might be needed for standalone testing, but Gunicorn should handle it.
chat_logger = logging.getLogger('chat') 

# === Enums ===

class OrderStatus(Enum):
    PENDING = "Pending"
    CONFIRMED = "Confirmed"
    CANCELLED = "Cancelled"
    COMPLETED = "Completed"

# === Data Classes ===

@dataclass
class OrderItem:
    item_id: str
    name: str
    quantity: int
    price: float # Price per unit
    customizations: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None
    status: str = 'confirmed' # Status of the individual item ('confirmed', 'needs_clarification', etc.)
    clarification_prompt: Optional[str] = None # Stores the question if status is 'needs_clarification'

    def __post_init__(self):
        if not isinstance(self.quantity, int) or self.quantity <= 0:
            raise ValueError("Quantity must be a positive integer")
        if not isinstance(self.price, (int, float)) or self.price < 0:
            raise ValueError("Price must be a non-negative number")

    @property
    def total_price(self) -> float:
        # Base price
        base = self.quantity * self.price
        # Add customization costs (assuming customizations dict might contain price adjustments)
        customization_cost = 0.0
        # Example: check for 'extras' which might be a list of dicts with 'price'
        if 'extras' in self.customizations and isinstance(self.customizations['extras'], list):
            for extra in self.customizations['extras']:
                if isinstance(extra, dict) and 'price' in extra:
                    try:
                        customization_cost += float(extra['price']) * self.quantity
                    except (ValueError, TypeError):
                        pass # Ignore if price is not a valid number
        return base + customization_cost

    def __repr__(self) -> str:
        prompt_str = f", prompt='{self.clarification_prompt}'" if self.clarification_prompt else ""
        return f"OrderItem(id={self.item_id}, name='{self.name}', qty={self.quantity}, price={self.price:.2f}, status='{self.status}'{prompt_str})"

@dataclass
class Order:
    order_id: str
    session_id: str
    items: List[OrderItem] = field(default_factory=list)
    status: OrderStatus = OrderStatus.PENDING
    total_amount: float = 0.0
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    notes: Optional[str] = None

    def update_total(self):
        """Recalculates the total amount based on items in the order."""
        self.total_amount = sum(item.total_price for item in self.items)
        self.updated_at = datetime.datetime.now()

    def add_item(self, item: OrderItem):
        """Adds an item to the order and updates the total."""
        # Optional: Check if item with same ID and customizations already exists, then increment quantity
        existing_item = next((i for i in self.items if i.item_id == item.item_id and i.customizations == item.customizations), None)
        if existing_item:
            existing_item.quantity += item.quantity
        else:
            self.items.append(item)
        self.update_total()

    def remove_item(self, item_id: str, quantity: int = 1):
        """Removes a specified quantity of an item by its ID."""
        # Find item (simple match by id for now)
        item_to_remove = next((i for i in self.items if i.item_id == item_id), None)
        if item_to_remove:
            if item_to_remove.quantity > quantity:
                item_to_remove.quantity -= quantity
            else:
                self.items.remove(item_to_remove)
            self.update_total()

    def update_status(self, new_status: OrderStatus):
        """Updates the order status."""
        self.status = new_status
        self.updated_at = datetime.datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Converts the order to a dictionary format suitable for JSON serialization."""
        items_as_dicts = []
        for item in self.items:
            item_dict = asdict(item)
            chat_logger.debug(f"[Order.to_dict] asdict(item) result: {item_dict}") # Log the dict
            items_as_dicts.append(item_dict)
            
        return {
            "order_id": self.order_id,
            "session_id": self.session_id,
            "items": items_as_dicts, # Use the list we built
            "status": self.status.value, 
            "total_amount": self.total_amount,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "notes": self.notes
        }

    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Creates an Order object from a dictionary."""
        items = [OrderItem(**item_data) for item_data in data.get('items', [])]
        status = OrderStatus(data.get('status', 'Pending')) # Default to Pending if missing/invalid
        created_at = datetime.datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.datetime.now()
        updated_at = datetime.datetime.fromisoformat(data['updated_at']) if 'updated_at' in data else datetime.datetime.now()

        return cls(
            order_id=data['order_id'],
            session_id=data['session_id'],
            items=items,
            status=status,
            total_amount=data.get('total_amount', 0.0),
            created_at=created_at,
            updated_at=updated_at,
            notes=data.get('notes')
        )
