import os
import shutil
import json

def rename_files(source_dir):
    """
    Renames all jpg files in the source directory with their Design IDs.

    Parameters:
    source_dir: str, the path to the directory containing the jpg files.

    Returns:
    None
    """
    
    # List all files in the source directory
    files = os.listdir(source_dir)
    
    for file in files:

        # Check if the file is a jpg
        if file.endswith('.jpg'):

            # Get the file extension
            _, ext = os.path.splitext(file)

            # Skip the first VL in the file name
            first_vl_index = file.find('VL')

            # Find the next VL in the file name
            if first_vl_index != -1:
                start_index = file.find('VL', first_vl_index + 2)
                if start_index != -1:
                    end_index = file.find('.', start_index)
                    new_name = file[start_index:end_index] if end_index != -1 else file[start_index:]

                    # Rename the file
                    original_file_path = os.path.join(source_dir, file)
                    new_file_path = os.path.join(source_dir, new_name + ext)
                    os.rename(original_file_path, new_file_path)

def organize_jpgs(source_dir):
    """
    For each jpg file in the source directory, creates a folder with the same name (excluding the .jpg extension)
    and moves the file into that folder.

    Parameters:
    source_dir: str, the path to the directory containing the jpg files.

    Returns:
    None
    """

    # List all files in the source directory
    files = os.listdir(source_dir)
    
    for file in files:
        # Check if the file is a jpg
        if file.endswith('.jpg'):

            # Create a folder name by removing the .jpg extension from the file name
            name, _ = os.path.splitext(file)
            folder_name = name
            folder_path = os.path.join(source_dir, folder_name)
            
            # Create the folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            
            # Move the file into the newly created folder
            shutil.move(os.path.join(source_dir, file), os.path.join(folder_path, file))

def create_metadata_in_folders(source_dir):
    """
    Goes through each subfolder in the given path and creates a metadata.json file with a specific blueprint.
    The 'design_label' field in the blueprint is replaced with the Deisgn ID.
    
    Parameters:
    source_dir: str, the path to the directory containing the subfolders.

    Returns:
    None
    """

    # Define the blueprint for the metadata.json content
    metadata_blueprint = {
        "image_url": "",
        "celebrity_name": "Test Model",
        "event": "Test",
        "design_label": "",
        "source_url": "https://maior.memorix.nl/collection-management/index/index/locale/en/verzameling/bbbd1b6a-df69-29d2-b80b-dbbe96b9b24c/limiet/10/template/default/entiteit/e7d9117b-b57e-5434-b0f3-c9fdc1dde3dd/uuid/5e36cb47-19ec-4c40-2ca5-5ce6c453bdc6",
        "timestamp": "2024-07-05T12:34:56Z"
    }
    
    # List all items in the source directory
    items = os.listdir(source_dir)
    
    for item in items:
        folder_path = os.path.join(source_dir, item)

        # Check if the item is a directory
        if os.path.isdir(folder_path):
            
            # Update the 'design_label' field with the formatted folder name
            metadata_blueprint['design_label'] = item

            # Define the path for the metadata.json file
            metadata_file_path = os.path.join(folder_path, 'metadata.json')
            
            # Write the updated blueprint to the metadata.json file
            with open(metadata_file_path, 'w') as metadata_file:
                json.dump(metadata_blueprint, metadata_file, indent=4)


if __name__ == '__main__':
    # Creates a folder for each jpg file in the source directory and moves the file into that folder
    # Also creates a metadata.json file in each folder with a specific blueprint
    # Processes a bunch of jpg files to organize them into a database-like structure

    source_directory = 'data/models'
    rename_files(source_directory)
    organize_jpgs(source_directory)
    create_metadata_in_folders(source_directory)
    