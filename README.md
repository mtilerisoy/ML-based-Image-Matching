# ML-based Image Matching

This project is designed to scrape images from Getty Images based on given keywords, process these images, and match the cloth designs to a reference database using CLIP model. The project includes scripts for scraping images, processing them, and managing metadata.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Scripts](#scripts)
  - [scrape.py](#scrapepy)
  - [main.py](#mainpy)
  - [utils.py](#utilspy)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Scraping Images

To scrape images from Getty Images based on a keyword, run the following command:

```sh
python ~/ML-based-Image-Matching/scripts/scrape.py --keyword "nature" --max_pages 10
```

You can also provide a CSV file with keywords:

```sh
python ~/ML-based-Image-Matching/scripts/scrape.py --csv_file "keywords.csv"
```

### Processing Images

To process the scraped images under `~/ML-based-Image-Matching/data/scraped/keyword/`, use the `main.py` script:

```sh
python ~/ML-based-Image-Matching/scripts/main.py
```

## Scripts

### scrape.py

This script scrapes images from Getty Images based on a given keyword and saves the images along with their metadata.

#### Functions

- `load_names_from_csv(file_path)`: Load keywords to scrape from a CSV file.
- `update_status_in_csv(file_path, keyword, new_status)`: Update the scraping status of a keyword in the CSV file.
- `get_images_from_page(keyword, page_num, family='creative')`: Scrapes image URLs and alt texts from a Getty Images search results page.
- `download_image_async(session, image_url, folder_name, keyword, image_id)`: Downloads an image from a given URL and saves it to a specified folder.
- `save_metadata_to_json(metadata, folder_name)`: Saves image metadata to a JSON file.
- `scrape_images_async(keyword, max_pages=200, family='creative')`: Scrapes images based on a keyword and saves them along with their metadata.

### main.py

This script processes the scraped images and performs various operations such as renaming directories and loading embeddings.

#### Functions

- `process_directory(subdir, scraped_images_dir, detected_dir, detected_metadata_path, CLIP_model, CLIP_transform, instance_seg_model, seg_processor, seg_model, design_embeddings, design_labels)`: Processes a directory of images to find matching images.

### utils.py

This script contains utility functions for image processing and metadata management.

#### Functions

- `open_and_convert_image(file_path)`: Opens an image file and converts it to a NumPy array.
- `load_design_embeddings(embeddings_path, labels_path=None)`: Loads design embeddings and optionally design labels from pickle files.
- `save_filtered_image(cropped_image_pil, data_dir, file, idx)`: Saves a filtered cropped image to a specified directory.
- `list_image_files(source_dir)`: Lists all files in a specified directory.
- `load_metadata(metadata_file_path)`: Loads metadata from a JSON file.
- `get_first_valid_subdirectory(folder_path)`: Gets the first valid subdirectory in a specified folder.
- `get_info(metadata, target_filename)`: Gets information about a target filename from metadata.

## Dependencies

The project requires the following dependencies:

- `aiohttp==3.8.5`
- `argparse`
- `asyncio`
- `beautifulsoup4==4.12.2`
- `bs4`
- `clip-anytorch==2.5.0`
- `matplotlib==3.8.0`
- `numpy==1.25.0`
- `Pillow==10.0.0`
- `pickle5==0.0.10`
- `requests==2.31.0`
- `shutil`
- `torch==2.0.1`
- `transformers==4.31.0`
- `ultralytics==8.0.106`

## Project Structure

```
.gitignore
data/
    detected/
        metadata.json
    embeddings/
    scraped/
README.md
requirements.txt
scripts/
    keywords.csv
    keywords2.csv
    main.py
    models.py
    processing.py
    scrape.py
    utils.py
```