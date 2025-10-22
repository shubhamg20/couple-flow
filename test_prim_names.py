#!/usr/bin/env python3
"""
Test script to show prim name mapping
"""

# Test the naming logic
OBJECTS = [
    "apple_csm", "banana_csm", "battery_csm", "blue_block_csm", "blue_cup_csm", 
    "blue_plate_csm", "carrot_csm", "corn_csm", "grapes_csm", "green_block_csm",
    "green_cup_csm", "green_tray_csm", "ice_cream_csm", "mug_csm", "pen_csm",
    "pink_plate_csm", "plastic_cup_csm", "red_block_csm", "red_bowl_csm", 
    "wood_block_csm", "yellow_bowl_csm"
]

print("Prim name mapping preview:")
print("="*60)

for object_name in OBJECTS:
    object_name_clean = object_name.replace('_csm', '')  # "banana_csm" -> "banana"
    
    # Convert to proper case for prim names
    if 'block' in object_name_clean:
        # Handle block naming: "blue_block" -> "BlueBlock"
        parts = object_name_clean.split('_')
        object_name_clean = ''.join(word.capitalize() for word in parts)
    elif 'cup' in object_name_clean:
        # Handle cup naming: "blue_cup" -> "BlueCup"
        parts = object_name_clean.split('_')
        object_name_clean = ''.join(word.capitalize() for word in parts)
    elif 'plate' in object_name_clean:
        # Handle plate naming: "blue_plate" -> "BluePlate"
        parts = object_name_clean.split('_')
        object_name_clean = ''.join(word.capitalize() for word in parts)
    elif 'bowl' in object_name_clean:
        # Handle bowl naming: "red_bowl" -> "RedBowl"
        parts = object_name_clean.split('_')
        object_name_clean = ''.join(word.capitalize() for word in parts)
    elif 'tray' in object_name_clean:
        # Handle tray naming: "green_tray" -> "GreenTray"
        parts = object_name_clean.split('_')
        object_name_clean = ''.join(word.capitalize() for word in parts)
    elif '_' in object_name_clean:
        # Handle other compound names: "ice_cream" -> "IceCream", "plastic_cup" -> "PlasticCup"
        parts = object_name_clean.split('_')
        object_name_clean = ''.join(word.capitalize() for word in parts)
    else:
        # Single word names: "apple" -> "Apple", "banana" -> "Banana"
        object_name_clean = object_name_clean.capitalize()
    
    print(f"{object_name:20} -> /{object_name_clean:15} -> /World/envs/env_.*/{object_name_clean}")
