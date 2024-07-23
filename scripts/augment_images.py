import os
import argparse
import numpy as np
from PIL import Image, ImageFilter, ImageChops
import imghdr

def rotate_image(image, degrees, save_path):
    """
    Rotates an image by a given number of degrees and saves the result.

    Parameters:
    - image: PIL.Image object, the image to rotate.
    - degrees: int or float, the angle to rotate the image by.
    - save_path: str, path where the rotated image will be saved.
    """

    # Rotate the image
    rotated_image = image.rotate(degrees)

    # Save the rotated image
    rotated_image.save(save_path[:-4]+'-rotated.jpg')

    print(f"Image rotated by {degrees} degrees and saved to {save_path}")


def blur_image(image, save_path):
    """
    Applies a blur effect to an image and saves the result.

    Parameters:
    - image: PIL.Image object, the image to blur.
    - save_path: str, path where the blurred image will be saved.
    """

    # Apply a blur effect to the image
    blurred_image = image.filter(ImageFilter.GaussianBlur())

    # Save the blurred image
    blurred_image.save(save_path[:-4]+'-blurred.jpg')

    print(f"Image blurred and saved to {save_path}")

def change_background_color(image, save_path):
    """
    Changes the background color of an image from white to a new color.

    Parameters:
    - image: PIL.Image object, the image to change the background color of.
    - save_path: str, path where the image with changed background color will be saved.
    """

    # Select a random RGB color
    new_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

    # Ensure image is in RGB mode to allow for easy manipulation
    image = image.convert("RGB")

    # Load the image data
    data = np.array(image)

    # Replace white or nearly white pixels with the new background color
    # Define white threshold in terms of RGB values
    white_threshold = 245

    # Create a mask where white or nearly white pixels are marked as True
    mask = np.all(data >= white_threshold, axis=-1)
    
    # Change all pixels in the mask to the new color
    data[mask] = new_color

    # Create a new image from the modified data
    result = Image.fromarray(data)

    # Save the image with the new background color
    result.save(save_path[:-4]+'-background.jpg')

    print(f"Background color changed and image saved to {save_path}")


if __name__ == "__main__":

    # Define the number of degrees to rotate the image by
    degrees = 30
    
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Rotate an image by a given number of degrees.")

    # Add the arguments
    parser.add_argument("root_dir", type=str, help="Path to the image folder.")

    # Parse the arguments
    args = parser.parse_args()
    path = args.root_dir

    # List all files in the source directory
    files = os.listdir(path)

    for file in files:
        # Get the path if the file is an image
        if file.endswith('.jpg'):
            image_path = os.path.join(path, file)

        # Check if the file is an image based on its extension and content
        if imghdr.what(image_path) is not None:
            try:
                # Open the image
                image = Image.open(image_path)
                # Rotate the image
                rotate_image(image, degrees, image_path)

                # Blur the image
                blur_image(image, image_path)

                # Background color change
                change_background_color(image, image_path)
            except IOError:
                print(f"Error opening image file: {image_path}")
        else:
            print(f"The file is not recognized as an image: {image_path}")

    