import os
from models import initialize_models
from utils import load_design_embeddings, load_metadata, list_image_files
from processing import process_file, copy_matched_files_and_update_metadata
# from memory_profiler import profile
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_directory(subdir, scraped_images_dir, detected_dir, detected_metadata_path, CLIP_model, CLIP_transform, instance_seg_model, seg_processor, seg_model, design_embeddings, design_labels):
    """
    Function to process a directory of scraped images.
    
    Parameters:
    - subdir: Subdirectory to process (str)
    - scraped_images_dir: Directory containing scraped images (str)
    - detected_dir: Directory to save detected images (str)
    - detected_metadata_path: Path to the detected metadata file (str)
    - CLIP_model: CLIP model
    - CLIP_transform: CLIP transform
    - instance_seg_model: Human Instance Segmentation model
    - seg_processor: Cloth Segmentation processor
    - seg_model: Cloth Segmentation model
    - design_embeddings: List of design embeddings
    - design_labels: List of design labels

    Returns:
    - None
    """
    subdir_path = os.path.join(scraped_images_dir, subdir)
    if os.path.isdir(subdir_path) and not subdir.startswith('x_'):
        print(f"Processing directory: {subdir_path}")

        # Load metadata and files
        metadata_file_path = os.path.join(subdir_path, "metadata.json")
        print(f"Loading metadata and files from {metadata_file_path}...")
        metadata = load_metadata(metadata_file_path)
        print(f"Metadata and files loaded. Length of metadata: {len(metadata)}")
        
        # Fixed variables
        sub_image_files = list_image_files(subdir_path)
        len_sub_files = len(sub_image_files)
        print(f"Length of sub files: {len_sub_files}")

        # Initialize variables to keep track of the number of matches and failed files
        match = 0

        print("Processing files...")
        for file_count, file in enumerate(sub_image_files):
            if file == ".DS_Store" or file == "metadata.json":
                continue
            print(f"Processing file {file_count}/{len_sub_files}: {file}")
            result = process_file(file, subdir_path, instance_seg_model, seg_processor, seg_model, design_embeddings, design_labels, CLIP_model, CLIP_transform)
            
            if result and result.match > 0:
                match += 1
                print(f"Match found for file: {file} | Match count: {match}/{len_sub_files}")

                # Create metadata_info dictionary
                metadata_info = {
                    "scraped_metadata": metadata,
                    "detected_metadata_file": detected_metadata_path,
                    "matched_scores": [result.best_score],
                    "matched_labels": result.top_k_design_labels
                }

                # Copy matched files to the detected directory and update metadata
                copy_matched_files_and_update_metadata(result.matched_files, subdir_path, detected_dir, metadata_info)

                # Clear intermediate results to free up memory
                del result

        # Rename the folder to label it as processed
        subdir_parent = os.path.dirname(subdir_path)
        subdir_name = os.path.basename(subdir_path)
        processed_dir = os.path.join(subdir_parent, f"x_{subdir_name}_processed")
        if not os.path.exists(processed_dir):
            os.rename(subdir_path, processed_dir)
        print(f"Renamed keyword directory to: {processed_dir}")

# @profile
def main():
    """
    Main function to process scraped images and detect designs.
    """
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Move one level up to the parent directory
        parent_dir = os.path.dirname(script_dir)

        # Construct the path to the 'data' folder
        source_dir = os.path.join(parent_dir, 'data')
        embeddings_dir = os.path.join(source_dir, "embeddings", "embeddings.pkl")
        labels_dir = os.path.join(source_dir, "embeddings", "labels.pkl")
        scraped_images_dir = os.path.join(source_dir, "scraped")
        detected_dir = os.path.join(source_dir, "detected")
        os.makedirs(detected_dir, exist_ok=True)
        detected_metadata_path = os.path.join(detected_dir, "metadata.json")

        print("Initializing models...")
        CLIP_model, CLIP_transform, instance_seg_model, seg_processor, seg_model = initialize_models()
        print("Models initialized.")

        print("Loading design embeddings...")
        design_embeddings, design_labels = load_design_embeddings(embeddings_dir, labels_dir)
        print("Design embeddings loaded.")

        # Get the list of subdirectories to process
        subdirs = [subdir for subdir in os.listdir(scraped_images_dir) if os.path.isdir(os.path.join(scraped_images_dir, subdir)) and not subdir.startswith('x_')]

        # Use ThreadPoolExecutor to process directories in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_directory, subdir, scraped_images_dir, detected_dir, detected_metadata_path, CLIP_model, CLIP_transform, instance_seg_model, seg_processor, seg_model, design_embeddings, design_labels) for subdir in subdirs]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred while processing a directory: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()