import sys
import argparse
import pickle

from helpers import load_image_from_folder



def main(folder_path):
    # Load the image from the subfolder
    image = load_image_from_folder(folder_path)

    # Create a dictionary with the image
    output_data = {
        "image": image,
    }

    # Serialize the dictionary and write it to stdout
    pickle.dump(output_data, sys.stdout.buffer)

    # return image

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Augment images in a folder')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing the scraped folders with images')
    args = parser.parse_args()

    # Get the path to the folder containing the scraped folders
    folder_path = args.folder_path

    main(folder_path)