import os
import json
from PIL import Image
import os

def load_image_from_folder(folder_path):
    """
    Loads an image as a PIL image from a given folder path.
    
    Parameters:
    folder_path: Path to the folder containing the image.

    Returns:
    A PIL Image object, or None if no image is found.
    """
    # List all files in the given folder
    files = os.listdir(folder_path)
    
    # Filter out image files based on common extensions
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    if image_files:
        # Assuming the first image file is the one we want to load
        image_path = os.path.join(folder_path, image_files[0])
        # Load and return the image
        return Image.open(image_path)
    
    # Return None if no image files are found
    return None

def load_design_label(metadata_path):
    """
    Loads the 'metadata.json' file from a given folder and returns the 'design_label' field.
    
    Parameters:
    metadata_path: Path to the 'metadata.json' file.

    Returns:
    The value of the 'design_label' field, or None if not found.
    """
    
    metadata_path = os.path.join(metadata_path, 'metadata.json')
    try:
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
            return metadata.get('design_label', None)
    except FileNotFoundError:
        print(f"File not found: {metadata_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {metadata_path}")
        return None

def find_folder(parent_dir, search_string):
    """
	Finds the folder that contains a given Design ID within a parent directory
	and retrieves the value of the 'design_label' field from the 'metadata.json' file inside this folder.
	
	Parameters:
	parent_dir: The parent directory to search within.
	search_string: The string to search for in folder names.
	
	Returns:
	The value of the 'design_label' field, or None if not found.
    """
	# Iterate over all items in the parent directory
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        
        # Check if the item is a directory and contains the search string
        if os.path.isdir(item_path) and search_string in item:
            return item_path
    
    # Return None if no matching folder or 'design_label' field is found
    return None

if __name__ == "__main__":
    parent_directory = 'data/models'
    search_str = 'VL65450'
    item_path = find_folder(parent_directory, search_str)
    print(f"Found folder: {item_path}")

    folder_path = item_path
    image = load_image_from_folder(folder_path)
    if image:
        image.show()  # This will display the image if it's loaded successfully
    else:
        print("No image found in the folder.")
    
    design_label = load_design_label(folder_path)
    print(f"Design label: {design_label}")
