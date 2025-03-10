import os
import numpy as np
import torch
import torchvision
import clip
import config
from PIL import Image
import matplotlib.pyplot as plt
import json
import shutil

def image_encoder(image: Image.Image, CLIP_MODEL, CLIP_TRANSFORM):
    """
    Function to encode an image using the CLIP model.
    
    Parameters:
    - image: Image to encode (PIL Image)
    - CLIP_MODEL: CLIP model
    - CLIP_TRANSFORM: CLIP transform
    
    Returns:
    - image_features: Encoded image features
    """
    image = CLIP_TRANSFORM(image).unsqueeze(0).to(config.device)
    with torch.no_grad():
        image_features = CLIP_MODEL.encode_image(image)
    return image_features


def crop_humans(image: Image, model=config.yolov8, show_images=False):
    """
    Function to segment and crop people in an image.
    
    Parameters:
    - image: np.ndarray, the image to segment and crop.
    - model: YOLO object, the YOLO model used for object detection.
    - show_images: bool, whether to display the cropped images or not.
    
    Returns:
    - cropped_images: list, the cropped images of people in the image.
    """

    # Run inference on a single image
    result = model(image, verbose=False)[0]

    # Process result
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for instance segmentation masks
    class_names = result.names  # Class names for the detected objects
    
    # Initialize the list of cropped images
    cropped_images = []

    # Loop through the boxes and masks and crop the people
    if masks is not None:
        for box in boxes:
            
            # Get the predicted class name
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

            # Skip if the class is not a person
            if class_name != 'person':
                # print(f"Skipping box {idx+1} as it is a {class_name}.")
                continue
            
            # Get coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop the masked image using the bounding box coordinates
            cropped_human = np.array(image)[y1:y2, x1:x2, :]

            cropped_images.append(cropped_human)
    else:
        print("No masks detected.")
    
    return cropped_images

def segment_clothes(image: Image.Image, seg_processor, seg_model):
    """
    Function to segment clothes from an image using a segmentation model.
    
    Parameters:
    - image: Image to segment clothes from (PIL Image)
    - SEGMENT_CLOTH_PROCESSOR: SegformerImageProcessor
    - SEGMENT_CLOTH_MODEL: SegformerForSemanticSegmentation
    
    Returns:
    - pred_seg: Segmented image
    """
    inputs = seg_processor(images=image, return_tensors="pt").to(config.device)
    outputs = seg_model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    mask = (pred_seg == 4) | (pred_seg == 5) | (pred_seg == 6) | (pred_seg == 7) | (pred_seg == 8) | (pred_seg == 16) | (pred_seg == 17)
    pred_seg[~mask] = 0
    pred_seg[mask] = 255
    return pred_seg

def segment_and_apply_mask(cropped_image: np.ndarray, seg_processor = config.seg_processor, seg_model = config.seg_model):
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

def calculate_similarity(cropped_image_pil: Image.Image, design_embeddings: list, design_labels: list, CLIP_model, CLIP_transform, k=5):
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
    # design_embeddings = torch.tensor(design_embeddings).to(config.device)
    image_features = image_encoder(cropped_image_pil, CLIP_model, CLIP_transform)
    image_features = image_features.to("cpu")
    similarities = [torch.nn.functional.cosine_similarity(image_features, t) for t in design_embeddings]
    similarities = torch.Tensor(similarities)
    
    # Sort the similarities and design labels in descending order
    sorted_similarities, _ = torch.sort(similarities, descending=True)
    # sorted_design_labels = [design_labels[i] for i in sorted_indices]
    
    # Calculate the average similarity score for the top-k designs
    avg_similarity = sorted_similarities[:k].mean().item()
    return avg_similarity # , sorted_similarities[:k], sorted_design_labels[:k]