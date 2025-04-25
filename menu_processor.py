import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    # Create handler
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    handler = logging.FileHandler(os.path.join(log_dir, "debug.log"), encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@dataclass
class MenuItem:
    id: str
    name: str
    price: float
    category: str
    description: Optional[str] = None
    customization_options: Dict[str, Dict[str, Union[str, List[str], Dict[str, str]]]] = None
    sizes: List[str] = None
    variants: List[str] = None
    allergens: List[str] = None

    def get_price_for_size(self, size: Optional[str] = None) -> float:
        """Get the price for a specific size, or default price if no size specified"""
        if not size or not self.customization_options or 'size' not in self.customization_options:
            return self.price
            
        # Get size options
        size_options = self.customization_options['size']
        default_size = size_options.get('default', 'Medium')
        
        # Size-based pricing
        if self.category == 'Drinks':
            size_prices = {
                'Small': self.price - 0.40,  # Small is $0.40 less than medium
                'Medium': self.price,        # Medium is base price
                'Large': self.price + 0.40,  # Large is $0.40 more than medium
                'One Size': self.price       # One size items use base price
            }
            return size_prices.get(size, self.price)
            
        return self.price

    def get_default_customizations(self) -> Dict[str, str]:
        """Get the default customizations for this item"""
        if not self.customization_options:
            return {}
            
        defaults = {}
        try:
            for category, options in self.customization_options.items():
                logger.debug(f"Processing defaults for category: {category}")
                logger.debug(f"Options data: {options}")
                if "default" in options:
                    defaults[category] = options["default"]
                elif "defaults" in options:
                    defaults[category] = options["defaults"]
            logger.debug(f"Final defaults: {defaults}")
        except Exception as e:
            logger.error(f"Error processing defaults for item: {self.name}")
            logger.error(f"Customization options: {self.customization_options}")
            logger.exception(e)
            raise
        return defaults

class MenuProcessor:
    def __init__(self):
        self.menu_items: Dict[str, MenuItem] = {}
        self.menu_text_chunks = []

    def process_menu(self, menu_file: str) -> int:
        logger.info(f"Processing menu file: {menu_file}")
        try:
            with open(menu_file, 'r') as f:
                menu_data = json.load(f)
                logger.debug(f"Loaded menu data: {menu_data.keys()}")

            # Process each menu item
            restaurant_data = menu_data.get('restaurant', {})
            logger.debug(f"Restaurant data keys: {restaurant_data.keys()}")
            
            menu = menu_data.get('menu', [])
            logger.debug(f"Found {len(menu)} menu categories")
            
            for category in menu:
                category_name = category.get('category', '')
                items = category.get('items', [])
                logger.debug(f"Processing category: {category_name} with {len(items)} items")
                
                for item in items:
                    try:
                        item_id = item['id']
                        logger.debug(f"Processing item: {item_id} - {item.get('name', 'NO NAME')}")
                        
                        # Extract customization options
                        customization_options = {}
                        if 'customizations' in item:
                            logger.debug(f"Found customizations for {item_id}: {item['customizations']}")
                            customization_options = item['customizations']

                        # Create MenuItem object
                        menu_item = MenuItem(
                            id=item_id,
                            name=item['name'],
                            price=float(item.get('price', 0)),
                            category=category_name,
                            description=item.get('description', ''),
                            customization_options=customization_options,
                            sizes=customization_options.get('size', {}).get('options', []) if customization_options and 'size' in customization_options else [],
                            variants=customization_options.get('variant', {}).get('options', []) if customization_options and 'variant' in customization_options else [],
                            allergens=item.get('allergens', [])
                        )
                        
                        self.menu_items[item_id] = menu_item
                        logger.debug(f"Successfully added menu item: {item_id}")

                        # Create searchable text chunk
                        chunk_text = self._create_menu_item_text(menu_item)
                        self.menu_text_chunks.append((chunk_text, item_id))

                    except Exception as e:
                        logger.error(f"Error processing menu item: {item.get('id', 'NO ID')} - {item.get('name', 'NO NAME')}")
                        logger.error(f"Item data: {item}")
                        logger.exception(e)
                        raise

            logger.info(f"Successfully processed {len(self.menu_items)} menu items")
            return len(self.menu_items)
            
        except Exception as e:
            logger.error("Error processing menu file")
            logger.error(f"Menu data: {menu_data}")
            logger.exception(e)
            raise

    def _create_menu_item_text(self, item: MenuItem) -> str:
        """Create a searchable text representation of a menu item"""
        try:
            text_parts = [
                f"Name: {item.name}",
                f"Price: ${item.price:.2f}",
                f"Category: {item.category}"
            ]
            
            if item.description:
                text_parts.append(f"Description: {item.description}")
                
            if item.customization_options:
                text_parts.append("Customization options:")
                for option_type, options in item.customization_options.items():
                    if 'options' in options:
                        text_parts.append(f"- {option_type}: {', '.join(str(opt) for opt in options['options'])}")
            
            if item.allergens:
                text_parts.append(f"Allergens: {', '.join(item.allergens)}")
                
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error creating menu item text for item: {item.name}")
            logger.error(f"Item data: {item}")
            logger.exception(e)
            raise

    def find_menu_items(self, query: str, current_order: Optional[Dict] = None) -> List[Tuple[str, str]]:
        """
        Find menu items relevant to the query, considering the current order context
        Returns a list of tuples (item_text, item_id)
        """
        try:
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
        except Exception as e:
            logger.error("Error finding menu items")
            logger.exception(e)
            raise

    def get_menu_item(self, item_id: str) -> Optional[MenuItem]:
        """Get a menu item by its ID"""
        try:
            logger.debug(f"Looking for menu item: {item_id}")
            logger.debug(f"Available menu items: {list(self.menu_items.keys())}")
            item = self.menu_items.get(item_id)
            if item:
                logger.debug(f"Found menu item: {item.name}")
            else:
                logger.debug(f"Menu item not found: {item_id}")
            return item
        except Exception as e:
            logger.error(f"Error getting menu item: {item_id}")
            logger.exception(e)
            raise

    def validate_customization(self, item_id: str, customizations: Dict[str, List[str]]) -> Tuple[bool, Optional[str]]:
        """
        Validate customization options for a menu item
        Returns (is_valid, error_message)
        """
        try:
            item = self.menu_items.get(item_id)
            if not item:
                return False, "Item not found"
            
            if not item.customization_options:
                if customizations:
                    return False, "This item doesn't support customization"
                return True, None
            
            for option_type, values in customizations.items():
                if option_type not in item.customization_options:
                    return False, f"Invalid customization type: {option_type}"
                
                # Get available options for this customization type
                options_data = item.customization_options[option_type]
                available_options = options_data.get('options', [])
                
                # Ensure values is a list
                if not isinstance(values, list):
                    return False, f"Expected list for {option_type}, got {type(values)}"
                
                # Check each value against available options
                for value in values:
                    if not any(value == (opt['name'] if isinstance(opt, dict) else opt) 
                             for opt in available_options):
                        return False, f"Invalid {option_type} option: {value}"
                
                # Check max selections if specified
                max_selections = options_data.get('max_selections', 3)
                if len(values) > max_selections:
                    return False, f"Too many {option_type} selections (max {max_selections})"
            
            # Check required customizations
            for opt_type, opt_data in item.customization_options.items():
                if opt_data.get('required', False) and opt_type not in customizations:
                    return False, f"Missing required {opt_type} customization"
            
            return True, None
        except Exception as e:
            logger.error(f"Error validating customization for item: {item_id}")
            logger.error(f"Customization options: {customizations}")
            logger.exception(e)
            raise

    def check_size_needed(self, item_id: str) -> bool:
        """Check if an item requires size selection"""
        try:
            item = self.menu_items.get(item_id)
            return item is not None and bool(item.sizes)
        except Exception as e:
            logger.error(f"Error checking size needed for item: {item_id}")
            logger.exception(e)
            raise

    def check_variant_needed(self, item_id: str) -> bool:
        """Check if an item requires variant selection"""
        try:
            item = self.menu_items.get(item_id)
            return item is not None and bool(item.variants)
        except Exception as e:
            logger.error(f"Error checking variant needed for item: {item_id}")
            logger.exception(e)
            raise

    def get_item_options(self, item_id: str) -> Dict[str, List[str]]:
        """Get all available options for an item"""
        try:
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
        except Exception as e:
            logger.error(f"Error getting item options for item: {item_id}")
            logger.exception(e)
            raise

processor = MenuProcessor()
