import os
import asyncio
from models import initialize_models
from utils import load_design_embeddings, load_metadata_and_files, get_first_valid_subdirectory, get_info
from processing import process_file
import json
import shutil

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Move one level up to the parent directory
    parent_dir = os.path.dirname(script_dir)

    # Construct the path to the 'data' folder
    source_dir = os.path.join(parent_dir, 'data')
    embeddings_dir = os.path.join(source_dir, "embeddings", "embeddings.pkl")
    labels_dir = os.path.join(source_dir, "embeddings", "labels.pkl")
    images_dir = os.path.join(source_dir, "scraped")
    detected_dir = os.path.join(source_dir, "detected")
    os.makedirs(detected_dir, exist_ok=True)
    detected_metadata_path = os.path.join(detected_dir, "metadata.json")
    convert_mode = "RGB"

    print("Initializing models...")
    CLIP_model, CLIP_transform, instance_seg_model, seg_processor, seg_model = initialize_models()
    print("Models initialized.")

    print("Loading design embeddings...")
    design_embeddings, design_labels = load_design_embeddings(embeddings_dir, labels_dir)
    print("Design embeddings loaded.")

    print("Getting first valid subdirectory...")
    keyword_dir = get_first_valid_subdirectory(images_dir)
    print(f"Keyword directory: {keyword_dir}")
    # Rename the folder to add 'x_' prefix
    keyword_dir_parent = os.path.dirname(keyword_dir)
    keyword_dir_name = os.path.basename(keyword_dir)
    new_keyword_dir = os.path.join(keyword_dir_parent, f"x_{keyword_dir_name}")
    os.rename(keyword_dir, new_keyword_dir)
    keyword_dir = new_keyword_dir
    print(f"Renamed keyword directory to: {keyword_dir}")

    # Load metadata and files
    metadata_file_path = os.path.join(keyword_dir, "metadata.json")
    print(f"Loading metadata and files from {metadata_file_path}...")
    metadata, sub_files = load_metadata_and_files(keyword_dir, metadata_file_path)
    detected_metadata = {"images": []}
    print("Metadata and files loaded.")

    match = 0
    ds_store_count = 0
    failed_files = []
    matched_files = []

    print("Processing files...")
    for file_count, file in enumerate(sub_files):
        if file == ".DS_Store" or file == "metadata.json":
            ds_store_count += 1
            continue
        print(f"Processing file {file_count + 1}/{len(sub_files)}: {file}")
        best_score, file_match, file_failed_files, file_matched_files, top_k_design_labels = process_file(file, keyword_dir, instance_seg_model, seg_processor, seg_model, design_embeddings, design_labels, metadata, CLIP_model, CLIP_transform)

        print(f"Matched files: {file_matched_files}")
        # for matched_file in file_matched_files:
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

            # matched_files.extend(file_matched_files[0])
            try:
                shutil.copy2(source_file_path, destination_file_path)
                print(f"Copied {file_matched_files[0]} to 'detected' directory.")
            except Exception as e:
                print(f"Failed to copy {file_matched_files[0]}: {e}")
            
        # if file_failed_files:
        #     failed_files.extend(file_failed_files)

        # Save the updated metadata.json
        try:
            with open(detected_metadata_path, "w", encoding="utf-8") as f:
                json.dump(detected_metadata, f, indent=4)
            print("metadata.json updated successfully in 'detected' directory.")
        except Exception as e:
            print(f"Failed to save metadata.json: {e}")
    

    print(f"Match: {match}/{len(sub_files)-ds_store_count}")
    print(f"Failed files: {failed_files}")
    print(f"Matched files: {matched_files}")

    # Rename the folder to label it as processed
    os.rename(keyword_dir, os.path.join(keyword_dir_parent, keyword_dir + "_processed"))
    print(f"Renamed keyword directory back to: {os.path.join(keyword_dir_parent, keyword_dir_name)}")


if __name__ == "__main__":
    main()
