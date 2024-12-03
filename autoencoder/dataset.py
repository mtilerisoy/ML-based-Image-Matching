# import os
# import torch
# from torch.utils.data import Dataset
# import cv2
# from concurrent.futures import ThreadPoolExecutor

# class AugmentedImageDataset(Dataset):
#     def __init__(self, config):
        
#         self.image_dir = config.image_dir
#         self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(".jp2")]
#         self.num_augmentations = config.num_augmentations
#         self.resize = config.image_size
#         self.num_workers = config.num_workers
#         self.images = self.load_images()

#     def load_image(self, image_path):
#         # print(image_path)
#         image = cv2.imread(image_path)
#         image = cv2.resize(image, self.resize)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
#         return image
    
#     def load_images(self):
#         with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
#             print("Loading images...")
#             print(f"Number of workers: {self.num_workers}")
#             image_paths = [os.path.join(self.image_dir, f) for f in self.image_files]
#             images = list(executor.map(self.load_image, image_paths))
#             print("Images loaded.")
#         return images

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         return self.images[idx]


import os
import torch
from torch.utils.data import Dataset
import cv2
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms
import random

class AugmentedImageDataset(Dataset):
    def __init__(self, config):
        self.image_dir = config.image_dir
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(".jp2")]
        self.num_augmentations = config.num_augmentations
        self.resize = config.image_size
        self.num_workers = config.num_workers
        self.images = self.load_images()

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.resize)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image
    
    def load_images(self):
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            print("Loading images...")
            print(f"Number of workers: {self.num_workers}")
            image_paths = [os.path.join(self.image_dir, f) for f in self.image_files]
            images = list(executor.map(self.load_image, image_paths))
            augmented_images = []
            for image in images:
                # Initialize a different set of augmentations for each image
                self.augmentations = self.get_augmentations()
                augmented_images.extend(self.augment_image(image))
            print("Images loaded.")
        return augmented_images
    
    def get_augmentations(self):
        
        # Random rotation angle between 15~45 and -45~-15 degrees
        flip_prob = random.random()

        if flip_prob > 0.5:
            rotation_angle = random.uniform(15, 45)
        else:
            rotation_angle = random.uniform(-45, -15)
            

        # Random factors for color jitter
        brightness = random.uniform(0.5, 1.5)
        contrast = random.uniform(0.5, 1.5)
        saturation = random.uniform(0.5, 1.5)
        hue = random.uniform(0.0, 0.1)

        # Random kernel size and sigma for Gaussian blur
        blur_kernel_size = random.choice([(3, 3), (5, 5), (7, 7)])
        blur_sigma = random.uniform(0.1, 2.0)

        return transforms.RandomRotation((rotation_angle, rotation_angle)), transforms.RandomVerticalFlip(), transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue), transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)                       

    def augment_image(self, image):
        augmented_images = [image]
        for augmentation in self.augmentations:
            augmented_image = augmentation(image)
            augmented_images.append(augmented_image)
        return augmented_images
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]