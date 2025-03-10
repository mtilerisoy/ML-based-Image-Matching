import torch
from PIL import Image
import numpy as np
import os
import time

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_human_seg():
    from ultralytics import YOLO
    
    # Load the YOLO model
    # Human Instance Segmentation model initialization
    model = YOLO("yolov8l-seg.pt")
    model = model.to(DEVICE)

    return model

def load_cloth_seg():
    from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

    # Cloth segmentation model initialization
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = model.to(DEVICE)

    return model, processor

def load_image_embed():
    from transformers import CLIPProcessor, CLIPModel

    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.to(DEVICE)

    return model, processor

def get_image_embedding(image, model, processor):

    # Process the image
    inputs = processor(images=image, return_tensors="pt")

    inputs.to(model.device)
    
    # Generate image embeddings
    with torch.no_grad():
        # image_embeddings = model.encode_image(inputs)
        image_embeddings = model.get_image_features(**inputs)
    return image_embeddings

def get_text_embedding(text, model, processor):
    # Preprocess the text
    inputs = processor(text=[text], return_tensors="pt")
    
    # Generate text embeddings
    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs)
    return text_embeddings

def crop_humans(image: Image, model):
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

            if class_name != 'person':
                continue # No person detected.
            
            # Get coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop the masked image using the bounding box coordinates
            cropped_human = np.array(image)[y1:y2, x1:x2, :]

            cropped_images.append(cropped_human)
    else:
        pass # No masks detected.
    
    return cropped_images

def segment_clothes(cropped_image: np.ndarray, model, processor):
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
    
    # Segment clothes from the image
    inputs = processor(images=cropped_image_pil, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=cropped_image_pil.size[::-1], mode="bilinear", align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    mask = (pred_seg == 4) | (pred_seg == 5) | (pred_seg == 6) | (pred_seg == 7) | (pred_seg == 8) | (pred_seg == 16) | (pred_seg == 17)
    pred_seg[~mask] = 0
    pred_seg[mask] = 255
    
    # Apply mask to filter out the background
    segmented_image = pred_seg.cpu().numpy().astype(np.uint8)
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

def segment_n_save_dir(folder_2_process, folder_2_save):
    # Load the models
    human_seg_model = load_human_seg()
    cloth_seg_model, cloth_seg_processor = load_cloth_seg()

    # Process the directory of images
    len_images = len(os.listdir(folder_2_process))
    print(f"Total number of images to process: {len_images}")

    for idx, filename in enumerate(os.listdir(folder_2_process)):
        if idx % 100 == 99:
            print(f"Processing image : {idx}/{len_images}")

        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.jp2')):
            image_path = os.path.join(folder_2_process, filename)
            image = Image.open(image_path).convert("RGB")

            # Crop the humans
            cropped_images = crop_humans(image, human_seg_model)

            for cropped_image in cropped_images:
                # Segment and apply mask to the cropped image
                segmented_image = segment_clothes(cropped_image, cloth_seg_model, cloth_seg_processor)
                
                # Check dimensions before saving
                if segmented_image.size[0] >= 120 and segmented_image.size[1] >= 120:
                    segmented_image.save(f"{folder_2_save}/image_{idx}.png")

# Main function to process images and save embeddings
def process_images_to_embeddings(image_dir, model, processor):
    # Initialize an empty tensor to store embeddings
    image_embeddings = None

    # Get list of images in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.jp2'))]
    len_images = len(image_files)

    # Process each image
    for idx, filename in enumerate(image_files):
        if idx % 10 == 0:
            print(f'Processing image {idx + 1} of {len_images}')
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        image_embedding = get_image_embedding(image, model, processor)

        # Stack embeddings
        if image_embeddings is None:
            image_embeddings = image_embedding
        else:
            image_embeddings = torch.vstack((image_embeddings, image_embedding))

    # Generate a unique filename based on the directory name and timestamp
    dir_name = os.path.basename(os.path.normpath(image_dir))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"{dir_name}_embeddings.pt"

    # Save the embeddings
    torch.save(image_embeddings, output_filename)
    print(f"Embeddings saved to {output_filename}")

    return image_embeddings

def rename_images_in_folder(folder_path):
    # Rename images in a folder

    for idx, filename in enumerate(os.listdir(folder_path)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.jp2')):
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, f"others_image_{idx}.png"))


if __name__ == "__main__":
    
    # Define the folders to process and save
    folder_2_process = "/Users/ilerisoy/Vlisco data/Week 16"
    folder_2_save =  "/Users/ilerisoy/Vlisco data/ACTUAL VLISCO/"
    
    try:
        segment_n_save_dir(folder_2_process, folder_2_save)
        # rename_images_in_folder(folder_2_save)
        # embed_n_save_dir(folder_2_process, folder_2_save)
    except Exception as e:
        print(f"Error: {e}")
        raise e