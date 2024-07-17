import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os


if __name__ == "__main__":

    # Ensure the 'cropped_images' directory exists
    output_dir = "cropped_images"
    os.makedirs(output_dir, exist_ok=True)

    # Load the YOLOv8 model with the updated 'u' model for improved performance
    model = YOLO('yolov10x.pt')  # Updated model path

    # Load an image
    image_path = 'data/model/x.jpg'
    image = Image.open(image_path)

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

        # Crop the image
        crop_img = image.crop((box[0], box[1], box[2], box[3]))

        # Save the cropped image
        crop_img.save(os.path.join(output_dir, f"{class_name}_{int(box[0])}_{int(box[1])}.jpg"))
    
    print("Cropping and saving complete.")