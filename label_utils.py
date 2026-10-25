"""
Phase 3: Label Mapping & Metadata Helper Module
Provides quick lookup for GTSRB 43 traffic sign categories, shapes, and types.
"""

import os
import json

# Primary 43 GTSRB Class Names
CLASS_NAMES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed & passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}

# Categorical groupings for deeper analysis and presentation
SIGN_CATEGORIES = {
    # Speed Limits
    0: 'Speed Limit', 1: 'Speed Limit', 2: 'Speed Limit', 3: 'Speed Limit',
    4: 'Speed Limit', 5: 'Speed Limit', 7: 'Speed Limit', 8: 'Speed Limit',
    
    # Prohibitory / Restrictions
    9: 'Prohibitory', 10: 'Prohibitory', 15: 'Prohibitory', 16: 'Prohibitory', 17: 'Prohibitory',
    
    # End of Limits / De-restriction
    6: 'De-restriction', 32: 'De-restriction', 41: 'De-restriction', 42: 'De-restriction',
    
    # Priority & Intersection
    11: 'Priority', 12: 'Priority', 13: 'Priority', 14: 'Priority',
    
    # Danger & Warning
    18: 'Danger / Warning', 19: 'Danger / Warning', 20: 'Danger / Warning', 21: 'Danger / Warning',
    22: 'Danger / Warning', 23: 'Danger / Warning', 24: 'Danger / Warning', 25: 'Danger / Warning',
    26: 'Danger / Warning', 27: 'Danger / Warning', 28: 'Danger / Warning', 29: 'Danger / Warning',
    30: 'Danger / Warning', 31: 'Danger / Warning',
    
    # Mandatory / Directional
    33: 'Mandatory', 34: 'Mandatory', 35: 'Mandatory', 36: 'Mandatory',
    37: 'Mandatory', 38: 'Mandatory', 39: 'Mandatory', 40: 'Mandatory'
}

# Sign Shapes (useful for computer vision feature explanation during viva)
SIGN_SHAPES = {
    14: 'Octagonal',
    13: 'Inverted Triangle',
    12: 'Diamond',
    **{i: 'Triangle' for i in range(18, 32)},
    **{i: 'Circle' for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]}
}

def get_label_name(class_id):
    """Returns readable sign name for a given class ID."""
    return CLASS_NAMES.get(int(class_id), f"Unknown Sign ({class_id})")

def get_category(class_id):
    """Returns sign broad category."""
    return SIGN_CATEGORIES.get(int(class_id), "Other")

def get_shape(class_id):
    """Returns geometric shape of sign."""
    return SIGN_SHAPES.get(int(class_id), "Unknown")

def get_full_metadata(class_id):
    """Returns dictionary with all metadata for a class."""
    cid = int(class_id)
    return {
        "class_id": cid,
        "name": get_label_name(cid),
        "category": get_category(cid),
        "shape": get_shape(cid)
    }

def export_metadata_json(output_path="class_metadata.json"):
    """Exports full metadata dictionary to a JSON file."""
    metadata = {str(cid): get_full_metadata(cid) for cid in range(43)}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return output_path

if __name__ == "__main__":
    export_metadata_json()
    print("Class metadata exported to 'class_metadata.json'.")
