# order_update_schema.py

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
                        "price": {"type": "number"},
                        "quantity": {"type": "integer", "default": 1},
                        "customizations": {"type": "object"},
                        "status": {
                            "type": "string", 
                            "enum": ["confirmed", "needs_customization"]
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
