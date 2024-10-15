import os
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import json
from utils import open_and_convert_image

def image_encoder(image, CLIP_MODEL, CLIP_TRANSFORM):
    model = CLIP_MODEL.eval()
    image = CLIP_TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features

def crop_humans(image, CROP_HUMAN_MODEL, show_images=False):
    assert isinstance(image, np.ndarray), "Image must be a numpy array."
    result = CROP_HUMAN_MODEL(image, verbose=False)[0]
    boxes = result.boxes
    masks = result.masks
    class_names = result.names
    cropped_images = []

    if masks is not None:
        for idx, (box, mask) in enumerate(zip(boxes, masks)):
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            if class_name != 'person':
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            mask_coords = mask.xy[0]
            blank_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(blank_mask, [mask_coords.astype(np.int32)], 1)
            masked_image = cv2.bitwise_and(image, image, mask=blank_mask)
            cropped_masked_img = masked_image[y1:y2, x1:x2]
            cropped_images.append(cropped_masked_img)
            if show_images:
                plt.imshow(cv2.cvtColor(cropped_masked_img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
    return cropped_images

def segment_clothes(image, SEGMENT_CLOTH_PROCESSOR, SEGMENT_CLOTH_MODEL):
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

def segment_and_apply_mask(cropped_image, seg_processor, seg_model):
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
    return cropped_image_pil, segmented_image_3ch

from dataclasses import dataclass
import os
import shutil

@dataclass
class ProcessResult:
    best_score: float
    match: int
    matched_files: list
    top_k_design_labels: list

def calculate_similarity(cropped_image_pil, design_embeddings, design_labels, CLIP_model, CLIP_transform, k=5):
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
    return avg_similarity, sorted_similarities[:k], sorted_design_labels

def process_cropped_images(cropped_images, seg_processor, seg_model, design_embeddings, design_labels, CLIP_model, CLIP_transform, threshold=0.72):
    """
    Processes a list of cropped images to find the best matching design based on similarity scores.

    Args:
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

    for idx, cropped_image in enumerate(cropped_images):
        try:
            cropped_image_pil, _ = segment_and_apply_mask(cropped_image, seg_processor, seg_model)
            if cropped_image_pil is None:
                continue
            avg_similarity, top_k_similarities, top_k_design_labels = calculate_similarity(cropped_image_pil, design_embeddings, design_labels, CLIP_model, CLIP_transform)
            if avg_similarity >= threshold:
                match += 1
            if avg_similarity > best_score:
                best_score = avg_similarity
        except Exception as e:
            print(f"Error processing file: {e}")
            continue
            
    return ProcessResult(best_score, match, [], top_k_design_labels)

def process_file(file, source_dir, instance_seg_model, seg_processor, seg_model, design_embeddings, design_labels, CLIP_model, CLIP_transform, threshold=0.72):
    """
    Processes a single file to find the best matching design based on similarity scores.

    Args:
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
    cropped_images = crop_humans(image_np, instance_seg_model, show_images=False)
    if not cropped_images:
        return None
    result = process_cropped_images(cropped_images, seg_processor, seg_model, design_embeddings, design_labels, CLIP_model, CLIP_transform, threshold)
    if result.match > 0:
        result.matched_files.append(file)
    return result

def copy_matched_files(matched_files, source_dir, target_dir, scraped_metadata, detected_metadata_file, matched_scores):
    """
    Copies matched files to a target directory.

    Args:
    - matched_files (list): List of matched file names.
    - source_dir (str): The source directory where the files are located.
    - target_dir (str): The target directory where the files will be copied.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for file in matched_files:
        shutil.copy(os.path.join(source_dir, file), target_dir)
        update_metadata(scraped_metadata, matched_files, detected_metadata_file, matched_scores)

def update_metadata(scraped_metadata, matched_files, detected_metadata_file, matched_scores):
    """
    Updates the metadata file with the matched files.

    Args:
    - scraped_metadata (dict): The scraped metadata dictionary.
    - matched_files (list): List of matched file names.
    - detected_metadata_file (str): The path to the detected metadata file.
    """
    # Load existing detected metadata
    if os.path.exists(detected_metadata_file):
        with open(detected_metadata_file, 'r') as file:
            detected_metadata = json.load(file)
    else:
        detected_metadata = {}

    # Update detected metadata with matched files
    for file, score in zip(matched_files, matched_scores):
        try:
            detected_metadata[file] = scraped_metadata[file]
            detected_metadata[file]['score'] = score
        except Exception as e:
            print(f"Error updating metadata: {e}")
            continue

    # Save updated detected metadata to the file
    with open(detected_metadata_file, 'w') as file:
        json.dump(detected_metadata, file, indent=4)
    print(f"Metadata updated and saved to {detected_metadata_file}")