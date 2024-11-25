import clip
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from ultralytics import YOLO

def initialize_models(device="cuda"):
    """
    Function to initialize the CLIP, YOLOv8 (Human Instance Detection), and Cloth Segmentation models.
    
    Returns:
    - CLIP_model: CLIP model
    - CLIP_transform: CLIP transform
    - instance_seg_model: Human Instance Segmentation model (YOLOv8)
    - seg_processor: Cloth Segmentation processor
    - seg_model: Cloth Segmentation model
    """
    # CLIP model initialization
    CLIP_model, CLIP_transform = clip.load("ViT-L/14@336px")
    CLIP_model = CLIP_model.to(device)

    # Human Instance Segmentation model initialization
    instance_seg_model = YOLO("yolov8l-seg.pt")
    instance_seg_model = instance_seg_model.to(device)

    # Cloth segmentation model initialization
    seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    seg_model = seg_model.to(device)

    return CLIP_model, CLIP_transform, instance_seg_model, seg_processor, seg_model