import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class MenuItem:
    id: str
    name: str
    price: float
    category: str
    description: Optional[str] = None
    customization_options: Dict[str, List[str]] = None
    sizes: List[str] = None
    variants: List[str] = None
    allergens: List[str] = None

class MenuProcessor:
    def __init__(self):
        self.menu_items: Dict[str, MenuItem] = {}
        self.menu_text_chunks = []

    def process_menu(self, menu_file: str) -> int:
        with open(menu_file, 'r') as f:
            menu_data = json.load(f)

        # Process each menu item
        for category in menu_data['restaurant']['menu']:
            category_name = category.get('category', '')
            for item in category.get('items', []):
                item_id = item['id']  # Use the ID directly from the JSON
                
                # Extract customization options
                customization_options = {}
                if 'customizations' in item:  
                    customization_options = item['customizations']

                # Create MenuItem object
                menu_item = MenuItem(
                    id=item_id,
                    name=item['name'],
                    price=float(item.get('price', 0)),
                    category=category_name,
                    description=item.get('description', ''),
                    customization_options=customization_options,
                    sizes=customization_options.get('size', {}).get('options', []) if customization_options else [],
                    variants=customization_options.get('variant', {}).get('options', []) if customization_options else [],
                    allergens=item.get('allergens', [])
                )
                
                self.menu_items[item_id] = menu_item

                # Create searchable text chunk
                chunk_text = self._create_menu_item_text(menu_item)
                self.menu_text_chunks.append((chunk_text, item_id))

        return len(self.menu_items)

    def _create_menu_item_text(self, item: MenuItem) -> str:
        """Create a searchable text representation of a menu item"""
        text_parts = [
            f"Item: {item.name}",
            f"Category: {item.category}",
            f"Price: ${item.price:.2f}"
        ]
        
        if item.description:
            text_parts.append(f"Description: {item.description}")
        
        if item.sizes:
            text_parts.append(f"Available sizes: {', '.join(item.sizes)}")
        
        if item.variants:
            text_parts.append(f"Available variants: {', '.join(item.variants)}")
        
        if item.customization_options:
            text_parts.append("Customization options:")
            for option_type, options in item.customization_options.items():
                text_parts.append(f"- {option_type}: {', '.join(options)}")
        
        if item.allergens:
            text_parts.append(f"Allergens: {', '.join(item.allergens)}")
        
        return "\n".join(text_parts)

    def find_menu_items(self, query: str, current_order: Optional[Dict] = None) -> List[Tuple[str, str]]:
        """
        Find menu items relevant to the query, considering the current order context
        Returns a list of tuples (item_text, item_id)
        """
        # Basic keyword matching for now
        query = query.lower()
        matches = []
        
        # First, check if this is a response to a pending clarification
        if current_order and 'items' in current_order:
            for item in current_order['items']:
                if item.get('status') != 'confirmed':
                    item_id = item.get('id')
                    if item_id in self.menu_items:
                        menu_item = self.menu_items[item_id]
                        matches.append((self._create_menu_item_text(menu_item), item_id))
                    break
        
        # Then look for new items in the query
        for text, item_id in self.menu_text_chunks:
            text_lower = text.lower()
            # Simple keyword matching
            if any(keyword in text_lower for keyword in query.split()):
                matches.append((text, item_id))
        
        return matches

    def get_menu_item(self, item_id: str) -> Optional[MenuItem]:
        """Get a menu item by its ID"""
        return self.menu_items.get(item_id)

    def validate_customization(self, item_id: str, customizations: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        """
        Validate customization options for a menu item
        Returns (is_valid, error_message)
        """
        item = self.menu_items.get(item_id)
        if not item:
            return False, "Item not found"
        
        if not item.customization_options:
            if customizations:
                return False, "This item doesn't support customization"
            return True, None
        
        for option_type, value in customizations.items():
            if option_type not in item.customization_options:
                return False, f"Invalid customization type: {option_type}"
            if value not in item.customization_options[option_type]:
                return False, f"Invalid {option_type} option: {value}"
        
        return True, None

    def check_size_needed(self, item_id: str) -> bool:
        """Check if an item requires size selection"""
        item = self.menu_items.get(item_id)
        return item is not None and bool(item.sizes)

    def check_variant_needed(self, item_id: str) -> bool:
        """Check if an item requires variant selection"""
        item = self.menu_items.get(item_id)
        return item is not None and bool(item.variants)

    def get_item_options(self, item_id: str) -> Dict[str, List[str]]:
        """Get all available options for an item"""
        item = self.menu_items.get(item_id)
        if not item:
            return {}
        
        options = {}
        if item.sizes:
            options['sizes'] = item.sizes
        if item.variants:
            options['variants'] = item.variants
        if item.customization_options:
            options['customization'] = item.customization_options
        
        return options

processor = MenuProcessor()
