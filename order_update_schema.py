# order_update_schema.py

from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel

class ItemStatus(str, Enum):
    CONFIRMED = "confirmed"
    NEEDS_CUSTOMIZATION = "needs_customization"
    PENDING = "pending"

class OrderItem(BaseModel):
    id: str
    name: str
    status: ItemStatus = ItemStatus.CONFIRMED
    clarification_needed: Optional[str] = None
    customizations: Dict[str, Union[str, List[str]]] = {}
    quantity: int = 1

class OrderUpdate(BaseModel):
    type: str = "order_update"
    items: List[OrderItem]
    message: str

order_update_schema = {
    "name": "order_update",
    "description": "Update to the user's order, including items and clarifications.",
    "parameters": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["order_update"]},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "quantity": {"type": "integer", "default": 1},
                        "customizations": {"type": "object"},
                        "status": {
                            "type": "string", 
                            "enum": ["confirmed", "needs_customization", "pending"]
                        },
                        "clarification_needed": {"type": "string"}
                    },
                    "required": ["id", "status"]
                }
            },
            "message": {"type": "string"}
        },
        "required": ["type", "items", "message"]
    }
}
