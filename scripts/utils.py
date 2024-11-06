import pickle
import os
import json
import numpy as np
from PIL import Image

import os

def get_first_valid_subdirectory(folder_path: str) -> str:
    """
    Function to get the first valid subdirectory in a specified folder.
    
    Parameters:
    - folder_path: Path to the folder (str)
    
    Returns:
    - sub_dir: Path to the first valid subdirectory (str)
    """
    subdirs = [subdir for subdir in os.listdir(folder_path) 
               if os.path.isdir(os.path.join(folder_path, subdir)) 
               and not subdir.startswith('x_') 
               and subdir != "DS_Store"]
    
    # if subdirs:
    #     return os.path.join(folder_path, subdirs[0])
    return subdirs

def open_and_convert_image(file_path: str) -> np.ndarray:
    """
    Function to open an image file and convert it to a NumPy array.
    
    Parameters:
    - file_path: Path to the image file (str)
    
    Returns:
    - image: Converted image as a NumPy array (np.ndarray)
    """
    image = Image.open(file_path).convert("RGB")
    return np.array(image)

def load_design_embeddings(embeddings_path: str, labels_path: str = None) -> tuple:
    """
    Function to load design embeddings and optionally design labels from pickle files.
    
    Parameters:
    - embeddings_path: Path to the embeddings pickle file (str)
    - labels_path: Path to the labels pickle file (str, optional)
    
    Returns:
    - design_embeddings: Loaded design embeddings (list)
    - design_labels: Loaded design labels (list)
    """
    with open(embeddings_path, 'rb') as f:
        design_embeddings = pickle.load(f)
    if labels_path:
        with open(labels_path, 'rb') as f:
            design_labels = pickle.load(f)
    else:
        design_labels = []
    return design_embeddings, design_labels

def save_filtered_image(cropped_image_pil: Image.Image, data_dir: str, file: str, idx: int):
    """
    Function to save a filtered cropped image to a specified directory.
    
    Parameters:
    - cropped_image_pil: Cropped image to save (PIL Image)
    - data_dir: Directory to save the image in (str)
    - file: Original file name (str)
    - idx: Index to append to the saved file name (int)
    """
    save_folder = os.path.join(data_dir, "cropped_images")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    cropped_image_pil.save(os.path.join(save_folder, f"{file[:-4]}_{idx}_filtered.jpg"))

def list_image_files(source_dir: str) -> list:
    """
    Function to list all files in a specified directory.
    
    Parameters:
    - source_dir: Directory to list files from (str)
    
    Returns:
    - sub_files: List of files in the directory (list)
    """
    sub_files = os.listdir(source_dir)
    return sub_files

def load_metadata(metadata_file_path: str) -> dict:
    """
    Function to load metadata from a JSON file.
    
    Parameters:
    - metadata_file_path: Path to the metadata JSON file (str)
    
    Returns:
    - metadata: Loaded metadata (dict)
    """
    with open(metadata_file_path, 'r') as f:
        return json.load(f)

def get_info(metadata: dict, target_filename: str) -> dict:
    """
    Function to get information about a target filename from metadata.
    
    Parameters:
    - metadata: Metadata dictionary (dict)
    - target_filename: Target filename to search for (str)
    
    Returns:
    - image_info: Information about the target filename (dict)
    """
    for image_info in metadata.get("images", []):
        if image_info.get("filename") == target_filename:
            return image_info

def check_design_label_match(top_k_design_labels: list, file: str) -> bool:
    """
    Function to check if any of the top-k design labels match the given file.
    
    Parameters:
    - top_k_design_labels: List of top-k design labels (list)
    - file: File name to check against (str)
    
    Returns:
    - match: True if a match is found, False otherwise (bool)
    """
    for design_label in top_k_design_labels:
        if design_label[:6] == file[:6]:
            return True
    return False