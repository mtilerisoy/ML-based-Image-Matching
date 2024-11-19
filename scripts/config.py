from sacred import Experiment
import os
import torch

import clip
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from ultralytics import YOLO, settings



ex = Experiment('Vlisco')

@ex.config
def mode_config():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    num_gpus = 1
    batch_size = 50
    num_workers = 4
    max_workers = 1
    seed = 42
    # threshold = 0.72
    threshold = 0.55

    # Set up the directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    embeddings_dir = os.path.join(data_dir, "embeddings", "embeddings.pkl")
    labels_dir = os.path.join(data_dir, "embeddings", "labels.pkl")
    scraped_images_dir = os.path.join(data_dir, "scraped")
    detected_dir = os.path.join(data_dir, "detected")
    # detected_metadata_path = os.path.join(detected_dir, "metadata.json")

    # Set up the model configurations
    model_config = {
        "clip_model": "ViT-L/14@336px",
        "yolov8": "utils/yolov8l-seg.pt",
        "seg_processor": "mattmdjaga/segformer_b2_clothes",
        "seg_model": "mattmdjaga/segformer_b2_clothes"
    }
    # CLIP_model, CLIP_transform = clip.load("ViT-L/14@336px")
    # CLIP_model = CLIP_model.to(device)
    # yolov8 = YOLO("utils/yolov8l-seg.pt")
    # yolov8 = yolov8.to(device)
    # seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    # seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    # seg_model = seg_model.to(device)

