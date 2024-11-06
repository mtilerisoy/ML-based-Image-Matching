from sacred import Experiment
import os


ex = Experiment('Vlisco')

@ex.config
def mode_config():
    device = "mps"
    num_gpus = 1
    batch_size = 1
    num_workers = 4
    max_workers = 1

    # Set up the directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    embeddings_dir = os.path.join(data_dir, "embeddings", "embeddings.pkl")
    labels_dir = os.path.join(data_dir, "embeddings", "labels.pkl")
    scraped_images_dir = os.path.join(data_dir, "scraped")
    detected_dir = os.path.join(data_dir, "detected")
    detected_metadata_path = os.path.join(detected_dir, "metadata.json")
