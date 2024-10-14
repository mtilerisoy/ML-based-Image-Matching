import os
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
# from utils import save_filtered_image_async, load_metadata_and_files_async, get_first_valid_subdirectory_async

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

def calculate_similarity(cropped_image_pil, design_embeddings, design_labels, CLIP_model, CLIP_transform):
    image_features = image_encoder(cropped_image_pil, CLIP_model, CLIP_transform)
    similarities = [torch.nn.functional.cosine_similarity(image_features, t) for t in design_embeddings]
    similarities = torch.Tensor(similarities)
    k = 5
    sorted_similarities = similarities.topk(k*2)
    sorted_design_labels = [design_labels[i] for i in sorted_similarities.indices]
    avg_similarity = sorted_similarities.values[:k].mean().item()
    return avg_similarity, sorted_similarities.values[:k], sorted_design_labels

def check_design_label_match(top_k_design_labels, file):
    for design_label in top_k_design_labels:
        if design_label[:6] == file[:6]:
            return True
    return False

def process_cropped_images(cropped_images, file, seg_processor, seg_model, design_embeddings, design_labels, metadata, CLIP_model, CLIP_transform):
    best_score = 0.0
    match = 0
    failed_files = []
    matched_files = []

    for idx, cropped_image in enumerate(cropped_images):
        cropped_image_pil, segmented_image_3ch = segment_and_apply_mask(cropped_image, seg_processor, seg_model)
        
        if cropped_image_pil is None:
            continue
        avg_similarity, top_k_similarities, top_k_design_labels = calculate_similarity(cropped_image_pil, design_embeddings, design_labels, CLIP_model, CLIP_transform)
        if avg_similarity > 0.719:
            matched_files.append(file)
            match += 1
        if avg_similarity > best_score:
            best_score = avg_similarity
            
    return best_score, match, failed_files, matched_files, top_k_design_labels

def process_file(file, source_dir, instance_seg_model, seg_processor, seg_model, design_embeddings, design_labels, metadata, CLIP_model, CLIP_transform):
    image_path = os.path.join(source_dir, file)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    cropped_images = crop_humans(image_np, instance_seg_model, show_images=False)
    if not cropped_images:
        return None, None, None, None, None
    best_score, match, failed_files, matched_files, top_k_design_labels = process_cropped_images(cropped_images, file, seg_processor, seg_model, design_embeddings, design_labels, metadata, CLIP_model, CLIP_transform)
    return best_score, match, failed_files, matched_files, top_k_design_labels

def process_matched_files(file_matched_files, keyword_dir, detected_dir, metadata, top_k_design_labels, best_score, detected_metadata_path):
    detected_metadata = {"images": []}
    
    # Load existing metadata if the file exists
    if os.path.exists(detected_metadata_path):
        with open(detected_metadata_path, "r", encoding="utf-8") as f:
            detected_metadata = json.load(f)
    
    if file_matched_files:
        # Initialize the source and destination paths to copy
        source_file_path = os.path.join(keyword_dir, file_matched_files[0])
        destination_file_path = os.path.join(detected_dir, file_matched_files[0])

        # Extract the metadata info
        image_info = get_info(metadata, file_matched_files[0])
        print(f"Type of image info: {type(image_info)}")

        # If the image info is empty, create a placeholder to update the score
        if image_info is None:
            image_info = {
                'filename': file_matched_files[0],
                'caption': "",
                'match': "true",
                'design': top_k_design_labels,
                'score': 0.0,
                'URL': ""
            }
        print(f"Type of image info: {type(image_info)}")
        
        # Update the score field
        image_info["score"] = best_score

        # Add the info to save later
        detected_metadata["images"].append(image_info)

        try:
            shutil.copy2(source_file_path, destination_file_path)
            print(f"Copied {file_matched_files[0]} to 'detected' directory.")
        except Exception as e:
            print(f"Failed to copy {file_matched_files[0]}: {e}")
        
    # Save the updated metadata.json
    try:
        with open(detected_metadata_path, "w", encoding="utf-8") as f:
            json.dump(detected_metadata, f, indent=4)
        print("metadata.json updated successfully in 'detected' directory.")
    except Exception as e:
        print(f"Failed to save metadata.json: {e}")