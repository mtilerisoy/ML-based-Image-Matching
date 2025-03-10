import pickle
import os
import json
import numpy as np
from PIL import Image

import os

def get_valid_subdirs(root_path: str) -> str:
    """
    Function to get the first valid subdirectory in a specified folder.
    
    Parameters:
    - root_path: Root path to the data folder (str)
    
    Returns:
    - sub_dir: Path to the first valid subdirectory (str)
    """
    subdirs = [subdir for subdir in os.listdir(root_path) 
               if os.path.isdir(os.path.join(root_path, subdir)) 
               and not subdir.startswith('x_') and not subdir.startswith('a_')
               and subdir != "DS_Store"]
    
    assert len(subdirs) > 0, f"No valid subdirectories found in {root_path}"
    return subdirs

def get_images_in_directory(directory: str) -> list:
    """
    Function to get the full path to all images in a specified directory.
    
    Parameters:
    - directory: Path to the directory (str)
    
    Returns:
    - image_paths: List of full paths to image files in the directory (list)
    """
    # Get all image files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png") or f.endswith(".jp2")]
    
    # Get the full path to each image file
    image_paths = [os.path.join(directory, f) for f in image_files]
    return image_paths

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

def save_metadata(metadata: dict, metadata_file_path: str):
    """
    Function to save metadata to a JSON file.
    
    Parameters:
    - metadata: Metadata to save (dict)
    - metadata_file_path: Path to the metadata JSON file (str)
    """
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def load_design_embeddings(name: str) -> tuple:
    """
    Function to load design embeddings and optionally design labels from pickle files.
    
    Parameters:
    - embeddings_path: Path to the embeddings pickle file (str)
    - labels_path: Path to the labels pickle file (str, optional)
    
    Returns:
    - design_embeddings: Loaded design embeddings (list)
    - design_labels: Loaded design labels (list)
    """
    if name == 'clip':
        with open("/Users/ilerisoy/Vlisco/ML-based-Image-Matching/data/embeddings/clip_embeddings.pkl", 'rb') as f:
            design_embeddings = pickle.load(f)
    
    elif name == 'vit':
        with open("/Users/ilerisoy/Vlisco/ML-based-Image-Matching/data/embeddings/vit_embeddings.pkl", 'rb') as f:
            design_embeddings = pickle.load(f)

    with open("/Users/ilerisoy/Vlisco/ML-based-Image-Matching/data/embeddings/labels.pkl", 'rb') as f:
        design_labels = pickle.load(f)

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

import os
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import pickle

def generate_embeddings():

    # Load pre-trained ViT model and feature extractor
    ViT_model_name = "google/vit-base-patch16-224"  # You can choose other ViT models
    ViT_model = ViTForImageClassification.from_pretrained(ViT_model_name, output_hidden_states=True)  # Enable hidden states
    ViT_feature_extractor = ViTFeatureExtractor.from_pretrained(ViT_model_name)

    # Move model to the appropriate device
    device = torch.device("mps")
    ViT_model.to(device)

    # Path to the directory containing the images
    images_dir = "/Users/ilerisoy/Vlisco data/Classics/designs"
    image_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith((".jpg", ".png", ".jpeg", ".jp2"))]

    # List to store the embeddings
    embeddings = []

    # Process each image
    for i, image_path in enumerate(image_paths):
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Preprocess the image using the ViT feature extractor
        inputs = ViT_feature_extractor(images=image, return_tensors="pt").to(device)
        
        # Extract features from the ViT model
        with torch.no_grad():
            outputs = ViT_model(**inputs)
            # Access the hidden states from the transformer layers
            hidden_states = outputs.hidden_states  # Tuple of all hidden states
            
            # Use the last hidden state (or average all hidden states) as features
            last_hidden_state = hidden_states[-1]  # Shape: [batch_size, sequence_length, hidden_size]
            
            # Average over the sequence dimension to get a single feature vector
            image_features = last_hidden_state.mean(dim=1)  # Shape: [batch_size, hidden_size]
        
        # Append the embedding to the list
        embeddings.append(image_features.cpu())  # Move to CPU and store
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} images")

    # Save the embeddings as a list of tensors using pickle
    output_path = "/Users/ilerisoy/Vlisco/ML-based-Image-Matching/data/embeddings/embeddings.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Embeddings saved to {output_path}")
