import clip
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from ultralytics import YOLO

def initialize_models():
    # CLIP model initialization
    CLIP_model, CLIP_transform = clip.load("ViT-L/14@336px")
    CLIP_model = CLIP_model.eval()

    # Human Instance Segmentation model initialization
    instance_seg_model = YOLO("yolov8l-seg.pt")

    # Cloth segmentation model initialization
    seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    return CLIP_model, CLIP_transform, instance_seg_model, seg_processor, seg_model