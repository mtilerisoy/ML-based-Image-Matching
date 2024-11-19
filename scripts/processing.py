import os
import numpy as np
import torch
import torchvision
import os
import configg
from PIL import Image
from time import sleep
from torchvision import transforms

CLIP_processor = transforms.Compose([
    transforms.Resize((336,336), interpolation=configg.BICUBIC),
    transforms.CenterCrop((336,336)),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

def process_batch(_config, batch, transform=None):

    assert isinstance(batch, list), "Batch must be a list of image paths."
    assert isinstance(batch[0], str), "Batch must be a list of image paths."

    # Load and transform images
    images = [transform(Image.open(image_path).convert("RGB")) for image_path in batch]
    images = torch.stack(images)

    # Crop the humans in the images
    batch_cropped_images = crop_humans(images)
    
    # Predict the labels
    labels = [predict_label(_config["threshold"], element) for element in batch_cropped_images]

    del images, batch_cropped_images
    # Free up memory
    if configg.DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif configg.DEVICE == "mps":
        torch.mps.empty_cache()
    
    return labels

def predict_label(threshold, batch_element):
    if batch_element[0].sum() == 0:
        return False
    else:
        # Encode the image
        image_features = image_encoder(batch_element, configg.CLIP_model, configg.CLIP_transform)
        # image_features = torch.randn(2, 768)
        # Compare the similarity
        for image in image_features:
            avg_similarity = calculate_similarity(image, configg.design_embeddings)
            # Return a label if there is a match
            if avg_similarity >= threshold:
                del image_features, image
                # Free up memory
                if configg.DEVICE == "cuda":
                    torch.cuda.empty_cache()
                elif configg.DEVICE == "mps":
                    torch.mps.empty_cache()
                return True
            else:
                del image_features, image
                # Free up memory
                if configg.DEVICE == "cuda":
                    torch.cuda.empty_cache()
                elif configg.DEVICE == "mps":
                    torch.mps.empty_cache()
                return False
    
    # Free up memory
    del image_features
    if configg.DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif configg.DEVICE == "mps":
        torch.mps.empty_cache()

def crop_humans(batch, model=configg.yolov8):
    """
    Function to segment and crop people in a batch of images.
    
    Parameters:
    - batch: torch.Tensor, the batch of images to segment and crop.
    - model: YOLO object, the YOLO model used for object detection.
    - save_images: bool, whether to save the cropped images or not.
    
    Returns:
    - all_cropped_images: list of lists, the cropped images of people in the batch.
    """

    # Assert the batch as a torch tensor
    assert isinstance(batch, torch.Tensor), "Batch must be a torch tensor."

    # Initialize the list of all cropped images
    cropped_images_list = []

    # Run inference on the batch
    results = model(batch.to(configg.DEVICE), verbose=False)
    print(f"Length of results: {len(results)}")

    # Process results list
    for idx, result in enumerate(results):
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for mask outputs

        if masks is None:
            # print(f"No masks detected for image {idx}.")
            cropped_images_list.append([torch.zeros((3, 320, 320), dtype=torch.float32)])
            continue
        
        cropped_images = []
        for i, box in enumerate(boxes):
            # Get the predicted class name
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            # Skip if the class is not a person
            if class_name != 'person':
                # print(f"Skipping box {idx+1} as it is a {class_name}.")
                continue

            # Get coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop the masked image using the bounding box coordinates
            masked_human = batch[idx, :, y1:y2, x1:x2].unsqueeze(0)

            # Resize the cropped image to 320x320
            masked_human = torch.nn.functional.interpolate(masked_human, size=(320, 320), mode="bilinear", align_corners=False)

            # Append the cropped image to the list
            cropped_images.append(masked_human.squeeze())

        # Stack the of cropped images to the batch_cropped_images list
        if cropped_images == []:
            cropped_images.append(torch.zeros((3, 320, 320), dtype=torch.float32))
        cropped_images_list.append(cropped_images)

        del boxes, masks, cropped_images
        if configg.DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif configg.DEVICE == "mps":
            torch.mps.empty_cache()
        
    # Free up memory
    del results
    if configg.DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif configg.DEVICE == "mps":
        torch.mps.empty_cache()

    return cropped_images_list

def image_encoder(batch_element: list, CLIP_MODEL, CLIP_TRANSFORM: torchvision.transforms.Compose):

    # Check if the input is a list of images
    assert isinstance(batch_element, list), "Input must be a list of images."

    all_image_features = []

    transformed_images = [CLIP_processor(cropped_image) for cropped_image in batch_element]
    transformed_images = torch.stack(transformed_images)

    with torch.no_grad():
        image_features = CLIP_MODEL.encode_image(transformed_images.to(configg.DEVICE))
        all_image_features.append(image_features)

    # Concatenate all image features
    if all_image_features:
        all_image_features = torch.cat(all_image_features, dim=0)
    else:
        all_image_features = torch.zeros(1, 768, device=configg.DEVICE)

    # Free up memory
    image_features = image_features.to("cpu")
    del transformed_images, image_features
    if configg.DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif configg.DEVICE == "mps":
        torch.mps.empty_cache()

    return all_image_features

def calculate_similarity(image_features: torch.Tensor, design_embeddings: list, k=5):
    image_features = image_features.cpu()
    similarities = [torch.nn.functional.cosine_similarity(image_features, t) for t in design_embeddings]
    similarities = torch.tensor(similarities)

    # Sort the similarities and design labels in descending order
    sorted_similarities, _ = torch.sort(similarities, descending=True)
    # sorted_design_labels = [design_labels[i] for i in sorted_indices]

    # Calculate the average similarity score for the top-k designs
    avg_similarity = sorted_similarities[:k].mean().item()

    del design_embeddings, similarities, sorted_similarities
    return avg_similarity # , sorted_similarities[:k], sorted_design_labels[:k]