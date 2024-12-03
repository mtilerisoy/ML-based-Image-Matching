data_dir = "/Users/ilerisoy/Vlisco data/Classics"
project_dir = "/Users/ilerisoy/Vlisco/Utils/"
image_dir = data_dir + "/" + "designs"
model_path = "model.pth"
save_model = True
models_dir = project_dir + "models/"
# save_path = image_dir + "/" + model_path

validation_split = 0.2
num_workers = 8
accelerator = "mps"
device = "mps"
num_augmentations = 2
epochs = 20
batch_size = 8
learning_rate = 1e-3
image_size = (336, 336)
