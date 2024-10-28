import os
import numpy as np
import torch
import torchvision
import clip
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import json
import shutil
from utils import open_and_convert_image
from ultralytics.models.yolo.model import YOLO
from transformers.models.segformer.image_processing_segformer import SegformerImageProcessor
from transformers.models.segformer.modeling_segformer import SegformerForSemanticSegmentation

def image_encoder(image: Image.Image, CLIP_MODEL: clip.model.CLIP, CLIP_TRANSFORM: torchvision.transforms.Compose):
    """
    Function to encode an image using the CLIP model.
    
    Parameters:
    - image: Image to encode (PIL Image)
    - CLIP_MODEL: CLIP model
    - CLIP_TRANSFORM: CLIP transform
    
    Returns:
    - image_features: Encoded image features
    """
    image = CLIP_TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        image_features = CLIP_MODEL.encode_image(image)
    return image_features

def crop_humans(image, CROP_HUMAN_MODEL, show_images=False):
    """
    Function to segment and crop people in an image.
    
    Parameters:
    - image: np.ndarray, the image to segment and crop.
    - model: YOLO object, the YOLO model used for object detection.
    - show_images: bool, whether to display the cropped images or not.
    
    Returns:
    - cropped_images: list, the cropped images of people in the image.
    """

    # Assert the image as numpy array
    assert isinstance(image, np.ndarray), "Image must be a numpy array."

    # Convert the image to RGB if it's in BGR format
    if image.shape[2] == 3 and np.array_equal(image[:, :, 0], image[:, :, 2]):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference on a single image
    result = CROP_HUMAN_MODEL(image, verbose=False)[0]

    # Process result
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for instance segmentation masks
    class_names = result.names  # Class names for the detected objects
    
    # Initialize the list of cropped images
    cropped_images = []

    # Loop through the boxes and masks and crop the people
    if masks is not None:
        for idx, (box, mask) in enumerate(zip(boxes, masks)):
            
            # Get the predicted class name
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

            # Skip if the class is not a person
            if class_name != 'person':
                print(f"Skipping box {idx+1} as it is a {class_name}.")
                continue
            
            # Get coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get the mask coordinates
            mask_coords = mask.xy[0]
            
            # Create a blank mask with the same dimensions as the original image
            blank_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Fill the blank mask with the mask coordinates
            cv2.fillPoly(blank_mask, [mask_coords.astype(np.int32)], 1)
            
            # Apply the mask to the original image
            masked_image = cv2.bitwise_and(image, image, mask=blank_mask)
            
            # Crop the masked image using the bounding box coordinates
            cropped_masked_img = masked_image[y1:y2, x1:x2]
            cropped_images.append(cropped_masked_img)
            
            if show_images:
                # Convert the image from BGR to RGB if necessary
                if cropped_masked_img.shape[2] == 3 and np.array_equal(cropped_masked_img[:, :, 0], cropped_masked_img[:, :, 2]):
                    cropped_masked_img_rgb = cv2.cvtColor(cropped_masked_img, cv2.COLOR_BGR2RGB)
                else:
                    cropped_masked_img_rgb = cropped_masked_img
                
                # Display the masked cropped image with the class name in the title
                plt.imshow(cropped_masked_img_rgb)
                plt.title(f'Masked Cropped Box {idx+1} - Class: {class_name}')
                plt.axis('off')
                plt.show()
    else:
        print("No masks detected.")
    
    return cropped_images

def segment_clothes(image: Image.Image, SEGMENT_CLOTH_PROCESSOR: SegformerImageProcessor, SEGMENT_CLOTH_MODEL: SegformerForSemanticSegmentation):
    """
    Function to segment clothes from an image using a segmentation model.
    
    Parameters:
    - image: Image to segment clothes from (PIL Image)
    - SEGMENT_CLOTH_PROCESSOR: SegformerImageProcessor
    - SEGMENT_CLOTH_MODEL: SegformerForSemanticSegmentation
    
    Returns:
    - pred_seg: Segmented image
    """
    inputs = SEGMENT_CLOTH_PROCESSOR(images=image, return_tensors="pt")
    outputs = SEGMENT_CLOTH_MODEL(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    mask = (pred_seg == 4) | (pred_seg == 5) | (pred_seg == 6) | (pred_seg == 7) | (pred_seg == 8) | (pred_seg == 16) | (pred_seg == 17)
    pred_seg[~mask] = 0
    pred_seg[mask] = 255
    return pred_seg

def segment_and_apply_mask(cropped_image: np.ndarray, seg_processor: SegformerImageProcessor, seg_model: SegformerForSemanticSegmentation):
    """
    Function to segment clothes from a cropped image and apply a mask to filter out the background.
    
    Parameters:
    - cropped_image: Cropped image to segment and apply mask to (NumPy array)
    - seg_processor: SegformerImageProcessor
    - seg_model: SegformerForSemanticSegmentation
    
    Returns:
    - cropped_image_pil: Cropped image with mask applied (PIL Image)
    """
    cropped_image_pil = Image.fromarray(cropped_image, mode='RGB')
    segmented_image = segment_clothes(cropped_image_pil, seg_processor, seg_model)
    segmented_image = segmented_image.cpu().numpy().astype(np.uint8)
    segmented_image_3ch = np.stack([segmented_image] * 3, axis=-1)
    filtered_image_np = np.where(segmented_image_3ch == 255, np.array(cropped_image_pil), 0)
    coords = np.column_stack(np.where(segmented_image == 255))
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped_image = filtered_image_np[y_min:y_max+1, x_min:x_max+1]
        cropped_image_pil = Image.fromarray(cropped_image, mode='RGB')
    else:
        cropped_image_pil = Image.fromarray(filtered_image_np, mode='RGB').convert("RGB")
    return cropped_image_pil

from dataclasses import dataclass
import os
import shutil

@dataclass
class ProcessResult:
    best_score: float
    match: int
    matched_files: list
    top_k_design_labels: list

def calculate_similarity(cropped_image_pil: Image.Image, design_embeddings: list, design_labels: list, CLIP_model: clip.model.CLIP, CLIP_transform: torchvision.transforms.Compose, k=5):
    """
    Function to calculate the similarity between the cropped image and the design embeddings and return the top k similar designs.

    Parameters:
    - cropped_image_pil: PIL image of the cropped human image
    - design_embeddings: List of design embeddings
    - design_labels: List of design labels
    - CLIP_model: CLIP model
    - CLIP_transform: CLIP transform
    - k: Number of top similar designs to return

    Returns:
    - avg_similarity: Average similarity score for top-k the design embeddings
    - sorted_similarities: Similarity scores in descending order
    - sorted_design_labels: Design labels corresponding to the similarity scores
    """
    image_features = image_encoder(cropped_image_pil, CLIP_model, CLIP_transform)
    similarities = [torch.nn.functional.cosine_similarity(image_features, t) for t in design_embeddings]
    similarities = torch.Tensor(similarities)
    
    # Sort the similarities and design labels in descending order
    sorted_similarities, sorted_indices = torch.sort(similarities, descending=True)
    sorted_design_labels = [design_labels[i] for i in sorted_indices]
    
    # Calculate the average similarity score for the top-k designs
    avg_similarity = sorted_similarities[:k].mean().item()
    return avg_similarity, sorted_similarities[:k], sorted_design_labels[:k]

def process_cropped_images(cropped_images: list, seg_processor: SegformerImageProcessor, seg_model: SegformerForSemanticSegmentation, design_embeddings: list, design_labels: list, CLIP_model: clip.model.CLIP, CLIP_transform: torchvision.transforms.Compose, threshold=0.72):
    """
    Processes a list of cropped images to find the best matching design based on similarity scores.

    Parameters:
    - cropped_images (list): List of cropped images to be processed.
    - seg_processor (object): The segmentation processor used to segment images.
    - seg_model (object): The segmentation model used to segment images.
    - design_embeddings (list): List of design embeddings for similarity comparison.
    - design_labels (list): List of labels corresponding to the design embeddings.
    - CLIP_model (object): The CLIP model used for calculating similarity.
    - CLIP_transform (object): The transformation applied to images before similarity calculation.
    - threshold (float): The similarity threshold for matching.

    Returns:
    - ProcessResult: An object containing the best score, match count, failed files, matched files, and top K design labels.
    """
    best_score = 0.0
    match = 0
    top_k_design_labels = []

    for idx, cropped_image in enumerate(cropped_images):
        try:
            cropped_image_pil = segment_and_apply_mask(cropped_image, seg_processor, seg_model)
            if cropped_image_pil is None:
                continue
            avg_similarity, _, top_k_design_labels = calculate_similarity(cropped_image_pil, design_embeddings, design_labels, CLIP_model, CLIP_transform)
            if avg_similarity >= threshold:
                match += 1
            if avg_similarity > best_score:
                best_score = avg_similarity
        except Exception as e:
            print(f"Error processing file: {e}")
            continue

    return ProcessResult(best_score, match, [], top_k_design_labels)

def process_file(file: str, source_dir: str, instance_seg_model: YOLO, seg_processor: SegformerImageProcessor, seg_model: SegformerForSemanticSegmentation, design_embeddings: list, design_labels: list, CLIP_model: clip.model.CLIP, CLIP_transform: torchvision.transforms.Compose, threshold=0.72):
    """
    Processes a single file to find the best matching design based on similarity scores.

    Parameters:
    - file (str): The file name to be processed.
    - source_dir (str): The directory where the file is located.
    - instance_seg_model (object): The instance segmentation model used to crop humans from the image.
    - seg_processor (object): The segmentation processor used to segment images.
    - seg_model (object): The segmentation model used to segment images.
    - design_embeddings (list): List of design embeddings for similarity comparison.
    - design_labels (list): List of labels corresponding to the design embeddings.
    - CLIP_model (object): The CLIP model used for calculating similarity.
    - CLIP_transform (object): The transformation applied to images before similarity calculation.
    - threshold (float): The similarity threshold for matching.

    Returns:
    - ProcessResult: An object containing the best score, match count, failed files, matched files, and top K design labels.
    - If no cropped images are found, returns None.
    """
    image_path = os.path.join(source_dir, file)
    image_np = open_and_convert_image(image_path)
    cropped_images = crop_humans(image_np, instance_seg_model)
    if not cropped_images:
        return None
    result = process_cropped_images(cropped_images, seg_processor, seg_model, design_embeddings, design_labels, CLIP_model, CLIP_transform, threshold)
    if result.match > 0:
        result.matched_files.append(file)
    return result

def copy_matched_files_and_update_metadata(matched_files: list, source_dir: str, target_dir: str, metadata_info: dict):
    """
    Copies matched files to a target directory and updates the metadata.

    Parameters:
    - matched_files (list): List of matched file names.
    - source_dir (str): The source directory where the files are located.
    - target_dir (str): The target directory where the files will be copied.
    - metadata_info (dict): Dictionary containing scraped metadata, detected metadata file path, matched scores, and matched labels.
    """
    scraped_metadata = metadata_info['scraped_metadata']
    detected_metadata_file = metadata_info['detected_metadata_file']
    matched_scores = metadata_info['matched_scores']
    matched_labels = metadata_info['matched_labels']

    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Load existing detected metadata
    if os.path.exists(detected_metadata_file):
        with open(detected_metadata_file, 'r') as file:
            detected_metadata = json.load(file)
    else:
        detected_metadata = {}

    # Copy files and update metadata
    for file, score in zip(matched_files, matched_scores):
        try:
            # Copy the file to the target directory
            shutil.copy(os.path.join(source_dir, file), target_dir)

            # Update the metadata
            detected_metadata[file] = scraped_metadata[file]
            detected_metadata[file]['score'] = score
            detected_metadata[file]['design'] = matched_labels

        except KeyError:
            print(f"File {file} not found in scraped metadata.")
        except Exception as e:
            print(f"Error updating metadata for file {file}: {e}")

    # Save updated detected metadata to the file
    with open(detected_metadata_file, 'w') as file:
        json.dump(detected_metadata, file, indent=4)
    print(f"Metadata updated and saved to {detected_metadata_file}")