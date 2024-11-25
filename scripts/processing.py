import os
import numpy as np
import torch
import torchvision
import os
import configg
from PIL import Image
from time import sleep
from torchvision import transforms
from matplotlib import pyplot as plt

CLIP_processor = transforms.Compose([
    transforms.Resize((336,336), interpolation=configg.BICUBIC),
    transforms.CenterCrop((336,336)),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

def load_batch(batch, transform):
    """
    Function to prepare a batch of images for processing.
    
    Parameters:
    - batch: list, the list of image paths.
    - transform: torchvision.transforms.Compose, the transformation to apply to the images.
    
    Returns:
    - images: torch.Tensor, the batch of images.
    """
    assert isinstance(batch, list), "Batch must be a list of image paths."
    assert isinstance(batch[0], str), "Batch must be a list of image paths."

    # Load and transform images
    images = [transform(Image.open(image_path).convert("RGB")) for image_path in batch]
    images = torch.stack(images)
    
    return images

def process_batch(batch, transform):

    # Load the batch of images
    images = load_batch(batch, transform)

    # Crop the humans in the images
    batch_cropped_images = crop_humans(images)
    
    # Predict the labels
    labels = [predict_label(configg.threshold, element) for element in batch_cropped_images]

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
            print(f"Average similarity: {avg_similarity}")
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
        print(f"Number of boxes detected in image {idx+1}: {len(result.boxes)}")
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for mask outputs

        if masks is None:
            print(f"No masks detected for image {idx}.")
            cropped_images_list.append([torch.zeros((3, 320, 320), dtype=torch.float32)])
            continue
        
        cropped_images = []
        for i, box in enumerate(boxes):
            # Get the predicted class name
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            # Skip if the class is not a person
            if class_name != 'person':
                print(f"Skipping box {i+1} in image {idx+1} as it is a {class_name}.")
                continue

            # Get coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop the masked image using the bounding box coordinates
            masked_human = batch[idx, :, y1:y2, x1:x2]

            # Resize the cropped image to 320x320
            masked_human = torch.nn.functional.interpolate(masked_human.unsqueeze(0), size=(320, 320), mode="bilinear", align_corners=False).squeeze()

            # save the masked_human to disk
            masked_human_pil = transforms.ToPILImage()(masked_human.cpu())
            masked_human_pil.save(os.path.join("filtered_images", f"cropped_human_{i}.png"))

            masked_human = segment_and_apply_mask(masked_human, configg.seg_processor, configg.seg_model)

            #plot the masked_human
            plt.imshow(masked_human)



            # Reshape the cropped image to [Channel, H, W]
            masked_human = torch.tensor(masked_human).permute(2, 0, 1)
            # Adjust the datatype
            masked_human = masked_human.to(torch.float32) / 255.0
            # Resize the cropped image to 320x320
            masked_human = torch.nn.functional.interpolate(masked_human.unsqueeze(0), size=(320, 320), mode="bilinear", align_corners=False).squeeze()

        
            # Append the cropped image to the list
            # cropped_images.append(masked_human.squeeze())
            cropped_images.append(torch.tensor(masked_human))

        

        # Stack the of cropped images to the batch_cropped_images list
        if cropped_images == []:
            cropped_images.append(torch.zeros((3, 320, 320), dtype=torch.float32))
        cropped_images_list.append(cropped_images)

        # save the cropped images to disk
        for i, cropped_image in enumerate(cropped_images):
            cropped_image_pil = transforms.ToPILImage()(cropped_image.cpu())
            cropped_image_pil.save(os.path.join("filtered_images", f"segemnted_image_{i}.png"))

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

def segment_clothes(images: torch.Tensor, SEGMENT_CLOTH_PROCESSOR, SEGMENT_CLOTH_MODEL):
    """
    Function to segment clothes from an image using a segmentation model.
    
    Parameters:
    - image: Image to segment clothes from (PIL Image)
    - SEGMENT_CLOTH_PROCESSOR: SegformerImageProcessor
    - SEGMENT_CLOTH_MODEL: SegformerForSemanticSegmentation
    
    Returns:
    - pred_seg: Segmented image
    """

    # image_tensors = torch.stack(images).to(configg.DEVICE)
    # images = images.to(configg.DEVICE)
    images.save(os.path.join("filtered_images", f"image_pil_{1}.png"))

    inputs = SEGMENT_CLOTH_PROCESSOR(images=images, return_tensors="pt").to(configg.DEVICE)
    outputs = SEGMENT_CLOTH_MODEL(**inputs)
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=images.size[::-1], mode="bilinear", align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    mask = (pred_seg == 4) | (pred_seg == 5) | (pred_seg == 6) | (pred_seg == 7) | (pred_seg == 8) | (pred_seg == 16) | (pred_seg == 17)

    pred_seg[~mask] = 0
    pred_seg[mask] = 255
    return pred_seg

    # # Plot the segmentation mask and input images side by side
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(pred_seg[0].cpu().numpy(), cmap="gray")
    # ax[0].set_title("Segmentation Mask")
    # ax[1].imshow(image_tensors[0].permute(1, 2, 0).cpu().numpy())
    # ax[1].set_title("Input Image")
    # plt.show()

    # # save the image_tensors to disk
    # for i in range(image_tensors.shape[0]):
    #     image_pil = Image.fromarray(image_tensors[i].permute(1, 2, 0).cpu().numpy())
    #     image_pil.save(os.path.join("filtered_images", f"image_tensor_{i}.png"))



    # # Save the segmentation mask and images to disk
    # for i in range(image_tensors.shape[0]):
    #     image_pil = image_tensors[i].permute(1, 2, 0).cpu().numpy()  # Shape: [H, W, C]
    #     pred_seg_pil = Image.fromarray(pred_seg[i].cpu().numpy())
    #     pred_seg_pil.save(os.path.join("filtered_images", f"segmented_image_{i}.png"))
    #     image_pil = Image.fromarray(image_pil)
    #     image_pil.save(os.path.join("filtered_images", f"image_{i}.png"))
    
    # # Filter the input images based on the segmentation mask
    # filtered_images = []
    # for i in range(image_tensors.shape[0]):
    #     # binary_mask = mask[i].unsqueeze(0).repeat(3, 1, 1)  # Shape: [C, H, W]
    #     # filtered_image = image_tensors[i] * binary_mask  # Apply the mask to the input image
    #     # filtered_images.append(filtered_image)  # Move the filtered image to CPU and add to the list

    #     filtered_image_np = np.where(pred_seg[i].cpu() == 255, np.array(image_tensors[i].cpu()), 0)
    #     coords = np.column_stack(np.where(pred_seg[i].cpu() == 255))
    #     if coords.size > 0:
    #         y_min, x_min = coords.min(axis=0)
    #         y_max, x_max = coords.max(axis=0)
    #         cropped_image = filtered_image_np[y_min:y_max+1, x_min:x_max+1]
    #         cropped_image_pil = Image.fromarray(cropped_image, mode='RGB')
    #     else:
    #         cropped_image_pil = Image.fromarray(filtered_image_np, mode='RGB').convert("RGB")
        
    #     # Save the filtered image to disk
    #     # filtered_image_np = filtered_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # Shape: [H, W, C]
    #     # filtered_image_pil = Image.fromarray(filtered_image_np)
    #     cropped_image_pil.save(os.path.join("filtered_images", f"filtered_image_{i}.png"))
    #     # filtered_image_pil.save(f"filtered_image_{i}.png")
    
    


def segment_and_apply_mask(cropped_image: torch.Tensor, seg_processor, seg_model):
    """
    Function to segment clothes from a cropped image and apply a mask to filter out the background.
    
    Parameters:
    - cropped_image: Cropped image to segment and apply mask to (NumPy array)
    - seg_processor: SegformerImageProcessor
    - seg_model: SegformerForSemanticSegmentation
    
    Returns:
    - cropped_image_pil: Cropped image with mask applied (PIL Image)
    """
    cropped_image_np = np.array(cropped_image)
    cropped_image_np = (cropped_image_np.transpose(1, 2, 0)*255).astype(np.uint8)
    cropped_image_pil = Image.fromarray(cropped_image_np)
    segmented_image = segment_clothes(cropped_image_pil, seg_processor, seg_model)
    segmented_image = segmented_image.cpu().numpy().astype(np.uint8)
    segmented_image_3ch = np.stack([segmented_image] * 3, axis=-1)
    filtered_image_np = np.where(segmented_image_3ch == 255, np.array(cropped_image_pil), 0)
    coords = np.column_stack(np.where(segmented_image == 255))
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped_image = filtered_image_np[y_min:y_max+1, x_min:x_max+1]

        # plot the cropped image
        plt.imshow(cropped_image)
        # cropped_image_pil = Image.fromarray(cropped_image, mode='RGB')
        return cropped_image
    else:
        # cropped_image_pil = Image.fromarray(filtered_image_np, mode='RGB').convert("RGB")
        plt.imshow(filtered_image_np)
        return filtered_image_np
    # return cropped_image_pil