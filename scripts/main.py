import os
import utils 
from processing import crop_humans, segment_and_apply_mask, calculate_similarity
from memory_profiler import profile
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
import torch
from PIL import Image
import shutil

def copy_matching_images(image_path, destination):
    image_name = os.path.basename(image_path)
    dest_path = os.path.join(destination, image_name)
    print(f"Copying {image_name} to {dest_path}")
    shutil.copy(image_path, dest_path)

def process_directory(images_list):

    count = 0
    for i in range(0, len(images_list)):
        count = count + 1
        image = Image.open(images_list[i]).convert("RGB")

        print(f"AT IMAGE INDEX: {i}")

        cropped_images = crop_humans(image, config.yolov8)
        
        for idx, cropped_image in enumerate(cropped_images):
            cropped_image_pil = segment_and_apply_mask(cropped_image, config.seg_processor, config.seg_model)
            if cropped_image_pil is None:
                continue
            avg_similarity = calculate_similarity(cropped_image_pil, config.design_embeddings, config.design_labels, config.CLIP_model, config.CLIP_transform)
            if avg_similarity >= config.threshold:
                # labels.append(images_list[i])
                copy_matching_images(images_list[i], config.detected_dir)
                break

        # Free up memory
        if config.DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif config.DEVICE == "mps":
            torch.mps.empty_cache()
        
        # copy_matching_images(labels, config.matching_images_dir)


@profile
def main():
    scraped_images_dir = config.scraped_images_dir

    subdirs = utils.get_valid_subdirs(scraped_images_dir)

    assert len(subdirs) > 0, f"No valid subdirectories found in {scraped_images_dir}"

    print(f"Number of subdirectories: {len(subdirs)}")

    
    for subdir in subdirs:
        images_list = utils.get_images_in_directory(os.path.join(scraped_images_dir, subdir))
        print(f"Number of images: {len(images_list)}")
        process_directory(images_list)

if __name__ == "__main__":
    main()