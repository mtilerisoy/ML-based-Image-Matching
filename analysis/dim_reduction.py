import os
import matplotlib.pyplot as plt
import torch

from preprocess_images import get_image_embedding, load_image_embed, process_images_to_embeddings

# Load the CLIP model and processor
model, processor = load_image_embed()

memorix_cropped_images_dir = '/Users/ilerisoy/Vlisco data/Data Folders Cropped/memorix'
website_cropped_images_dir = '/Users/ilerisoy/Vlisco data/Data Folders Cropped/website_photoshoot'
vintage_model_cropped_images_dir = '/Users/ilerisoy/Vlisco data/Data Folders Cropped/vintage_model_photoshoot'
fashion_images_0 = "/Users/ilerisoy/Vlisco data/Data Folders Cropped/fashion_images_0_short_short/"
fashion_images_1 = "/Users/ilerisoy/Vlisco data/Data Folders Cropped/fashion_images_1_short/"
possible_vlisco = "/Users/ilerisoy/Vlisco data/Data Folders Cropped/possible_vlisco/all"
not_vlisco = "/Users/ilerisoy/Vlisco data/Data Folders Cropped/not_vlisco/"
masked_designs = "/Users/ilerisoy/Vlisco data/Masked Design Images Short/"

# Process images and save embeddings
memorix_image_embeddings = process_images_to_embeddings(memorix_cropped_images_dir, model, processor)
website_image_embeddings = process_images_to_embeddings(website_cropped_images_dir, model, processor)
vintage_image_embeddings = process_images_to_embeddings(vintage_model_cropped_images_dir, model, processor)
fashion_image_0_embeddings = process_images_to_embeddings(fashion_images_0, model, processor)
fashion_image_1_embeddings = process_images_to_embeddings(fashion_images_1, model, processor)
possible_vlisco_embeddings = process_images_to_embeddings(possible_vlisco, model, processor)
not_vlisco_embeddings = process_images_to_embeddings(not_vlisco, model, processor)

masked_designs_embeddings = process_images_to_embeddings(masked_designs, model, processor)


print("Memorix embeddings shape:", memorix_image_embeddings.shape)
print("Website embeddings shape:", website_image_embeddings.shape)
print("Vintage model embeddings shape:", vintage_image_embeddings.shape)
print("Fashion images 0 embeddings shape:", fashion_image_0_embeddings.shape)
print("Fashion images 1 embeddings shape:", fashion_image_1_embeddings.shape)
print("Possible Vlisco embeddings shape:", possible_vlisco_embeddings.shape)
print("Not Vlisco embeddings shape:", not_vlisco_embeddings.shape)

import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np

# Example embedding tensors and labels (replace with your actual data)
# all_embeddings is a stacked tensor of embeddings from multiple datasets
# all_labels is a numpy array of labels indicating which dataset each embedding belongs to

embedding_tensors = [not_vlisco_embeddings, fashion_image_0_embeddings, memorix_image_embeddings, vintage_image_embeddings, website_image_embeddings, possible_vlisco_embeddings]
dataset_names = ['Not Vlisco', 'Plain Images', 'Memorix', 'Vintage', 'Website', 'Possible Vlisco']

# Combine all embeddings
all_embeddings = torch.vstack(embedding_tensors)

# Create labels for each dataset
all_labels = np.concatenate([
    np.full(embedding.shape[0], idx) for idx, embedding in enumerate(embedding_tensors)
])

# Apply UMAP
umap_reducer = umap.UMAP(random_state=42)
embeddings_2d_umap = umap_reducer.fit_transform(all_embeddings.cpu().numpy())

# Save the UMAP model
torch.save(umap_reducer, 'umap_model_original.pt')

# umap_reducer = torch.load('umap_model_original.pt')
# embeddings_2d_umap = umap_reducer.transform(all_embeddings.cpu().numpy())

# Define colors for each dataset
colors = ['black', 'red', 'blue', 'green', 'orange', 'cyan', 'purple', 'yellow', 'magenta', 'pink']

# Plot the results with color information
plt.figure(figsize=(10, 8))
for label, color in zip(np.unique(all_labels), colors):
    plt.scatter(
        embeddings_2d_umap[all_labels == label, 0],
        embeddings_2d_umap[all_labels == label, 1],
        c=color,
        label=f'{dataset_names[label]}',
        s=5,
        alpha=0.6
    )
plt.title('UMAP Projection of the Embeddings')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend()
plt.show()