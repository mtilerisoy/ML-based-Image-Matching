import os
from models import initialize_models
from utils import load_design_embeddings, load_metadata_and_images, get_first_valid_subdirectory
from processing import process_file, copy_matched_files_and_update_metadata

def main():
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

        print("Getting first valid subdirectory...")
        image_dir = get_first_valid_subdirectory(scraped_images_dir)
        print(f"Keyword directory: {image_dir}")
        

        # Load metadata and files
        metadata_file_path = os.path.join(image_dir, "metadata.json")
        print(f"Loading metadata and files from {metadata_file_path}...")
        metadata, sub_image_files = load_metadata_and_images(image_dir, metadata_file_path)
        print(f"Metadata and files loaded. Length of metadata: {len(metadata)}")
        
        # Fixed variables
        len_sub_files = len(sub_image_files)
        print(f"Length of sub files: {len_sub_files}")

        # Rename the folder to add 'x_' prefix
        image_dir_parent = os.path.dirname(image_dir)
        image_dir_name = os.path.basename(image_dir)
        new_keyword_dir = os.path.join(image_dir_parent, f"x_{image_dir_name}")
        if not os.path.exists(new_keyword_dir):
            os.rename(image_dir, new_keyword_dir)
        image_dir = new_keyword_dir
        print(f"Renamed keyword directory to: {image_dir}")

        # Initialize variables to keep track of the number of matches and failed files
        match = 0
        matched_files = []
        matched_scores = []

        print("Processing files...")
        for file_count, file in enumerate(sub_image_files):
            if file == ".DS_Store" or file == "metadata.json":
                continue
            print(f"Processing file {file_count}/{len_sub_files}: {file}")
            result = process_file(file, image_dir, instance_seg_model, seg_processor, seg_model, design_embeddings, design_labels, CLIP_model, CLIP_transform)
            
            if result and result.match > 0:
                matched_files.extend(result.matched_files)
                matched_scores.append(result.best_score)
                match += 1
                print(f"Match found for file: {file} | Match count: {match}/{len_sub_files}")

        # Create metadata_info dictionary
        metadata_info = {
            "scraped_metadata": metadata,
            "detected_metadata_file": detected_metadata_path,
            "matched_scores": matched_scores,
            "matched_labels": design_labels
        }

        # Copy matched files to the detected directory and update metadata
        copy_matched_files_and_update_metadata(matched_files, image_dir, detected_dir, metadata_info)

        # Rename the folder to label it as processed
        processed_dir = os.path.join(image_dir_parent, f"x_{image_dir_name}_processed")
        if not os.path.exists(processed_dir):
            os.rename(image_dir, processed_dir)
        print(f"Renamed keyword directory to: {processed_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()