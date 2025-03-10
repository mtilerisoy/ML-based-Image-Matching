from models import initialize_models
import utils
import torch
from PIL import Image
import os

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Initialize the models
CLIP_model, CLIP_transform, yolov8, seg_processor, seg_model = initialize_models(DEVICE)

# Load the design embeddings and labels
design_embeddings, design_labels = utils.load_design_embeddings()

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
num_gpus = 1
batch_size = 50
num_workers = 4
max_workers = 1
seed = 42
# threshold = 0.72
threshold = 5 # 9
# threshold = 0.55

# Set up the directory paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')
embeddings_dir = os.path.join(data_dir, "embeddings", "embeddings.pkl")
labels_dir = os.path.join(data_dir, "embeddings", "labels.pkl")
scraped_images_dir = os.path.join(data_dir, "scraped")
detected_dir = os.path.join(data_dir, "detected")
copy_dir = None
