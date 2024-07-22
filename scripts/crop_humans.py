import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

from helpers import find_folder

def crop_image(image, model):
    """
    Crops an image by detecting humans using a YOLO model.

    Parameters:
    - image: PIL.Image object, the image to crop.
    - model: YOLO object, the YOLO model to use for detection.

    Returns:
    - PIL.Image object, the cropped image or None if no humans are detected.
    """

    # Resize the image to make its dimensions divisible by 32
    new_width = (image.width // 32) * 32
    new_height = (image.height // 32) * 32
    image = image.resize((new_width, new_height))

    # Convert the image to a tensor
    img = torch.from_numpy(np.array(image)).float()
    img /= 255.0  # Normalize the image
    img = img.permute((2, 0, 1)).unsqueeze(0)  # Add batch dimension

    # Inference
    results = model(img)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    classes = results[0].boxes.cls.cpu().tolist()
    
    for box, cls in zip(boxes, classes):
        # Get the class name from the model class dictionary
        class_name = model.names[cls]

        if class_name == 'person':
            # Crop the image
            crop_img = image.crop((box[0], box[1], box[2], box[3]))
            return crop_img
        else:
            return None


def crop_and_save_images(source_dir, model_path='yolov10x.pt'):
    """
    Crops images detected by YOLO model and saves them to a specified directory.

    Parameters:
    - source_dir: str, Path to the input image.
    - model_path: str, Path to the YOLO model.
    """

    # Load the YOLO model
    model = YOLO(model_path)

    # Get the path to the image
    files = os.listdir(source_dir)
    for file in files:
        # Check if the file is a jpg
        if file.endswith('.jpg'):
            name, _ = os.path.splitext(file)
            image_path = os.path.join(source_dir, file)
            print(f"Processing {file}")

            # Load an image
            image = Image.open(image_path)

            # Crop the image
            crop_img = crop_image(image, model)

            # Save the cropped image
            crop_img.save(os.path.join(source_dir, f"{name}_person.jpg"))
            
            print("Cropping and saving complete.")

if __name__ == "__main__":

    source_dir = find_folder('data/models', 'VL00562')
    print(f"Found folder: {source_dir}")
    
    crop_and_save_images(source_dir)