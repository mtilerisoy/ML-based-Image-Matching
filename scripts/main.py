import os
import utils 
from processing import process_batch # , predict_label, crop_humans
from memory_profiler import profile
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import ex
import torch
from torchvision import transforms
from time import sleep
from PIL import Image
import configg

def process_directory(subdir, transform):
    images_list = utils.get_images_in_directory(os.path.join(configg.scraped_images_dir, subdir))
    batch_size = configg.batch_size

    labels = []
    count = 0
    for i in range(0, len(images_list), batch_size):
        count = count + 1
        batch = images_list[i:i+batch_size]

        print(f"AT INDEX: {i}\nBATCH: {batch}")
        
        labels.extend(process_batch(batch, transform))
        
        # Free up memory
        if configg.DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif configg.DEVICE == "mps":
            torch.mps.empty_cache()

        if count == 5:
            break
            raise SystemExit("Processed a single batch")
    
    for idx, label in enumerate(labels):
        if label:
            print(images_list[idx])


@profile
@ex.automain
def main():
    scraped_images_dir = configg.scraped_images_dir

    subdirs = utils.get_valid_subdirs(scraped_images_dir)

    assert len(subdirs) > 0, f"No valid subdirectories found in {scraped_images_dir}"

    # Define the transform to convert images to tensors
    transform = transforms.Compose([
        transforms.Resize((320, 320)),  # Resize images to a fixed size
        transforms.ToTensor()  # Convert images to tensors
    ])

    for subdir in subdirs:
        images_list = utils.get_images_in_directory(os.path.join(scraped_images_dir, subdir))
        print(f"Number of images: {len(images_list)}")
        process_directory(subdir, transform)

# TODO: Implement cloth instance segmentation
    # TODO: add the functionality under crop_humans where you mask the humans and then crop the images
    ######: Check the list/tensor structure of the cropped_images and