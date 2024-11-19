from models import initialize_models
import utils
import torch
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Initialize the models
CLIP_model, CLIP_transform, yolov8, seg_processor, seg_model = initialize_models(DEVICE)

# Load the design embeddings and labels
# design_embeddings, design_labels = utils.load_design_embeddings()
design_embeddings = utils.load_design_embeddings()
