import pickle
import os
import json
import numpy as np
from PIL import Image

def load_design_embeddings(embeddings_path, labels_path=None):
    with open(embeddings_path, 'rb') as f:
        design_embeddings = pickle.load(f)
    if labels_path:
        with open(labels_path, 'rb') as f:
            design_labels = pickle.load(f)
    else:
        design_labels = []
    return design_embeddings, design_labels

def save_filtered_image(cropped_image_pil, data_dir, file, idx):
    save_folder = os.path.join(data_dir, "cropped_images")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    cropped_image_pil.save(os.path.join(save_folder, f"{file[:-4]}_{idx}_filtered.jpg"))

def load_metadata_and_files(source_dir, metadata_file_path):
    with open(metadata_file_path, 'r') as f:
        metadata = json.load(f)
    sub_files = os.listdir(source_dir)
    return metadata, sub_files

def load_metadata(metadata_file_path):
    with open(metadata_file_path, 'r') as f:
        return json.load(f)

def get_first_valid_subdirectory(folder_path):
    for sub_dir in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, sub_dir)) and not sub_dir.startswith('x_') and sub_dir != "DS_Store":
            return os.path.join(folder_path, sub_dir)
    return None

def get_info(metadata, target_filename):
    # Search for the target filename in the images list
    for image_info in metadata.get("images", []):
        if image_info.get("filename") == target_filename:
            return image_info