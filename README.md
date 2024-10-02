# ML-based-Image-Matching

# Image Scraper Script

This script scrapes images from Getty Images based on a given keyword and saves the images along with their metadata. It can load keywords from a CSV file, update their status, and handle image downloading and metadata storage.

## Dependencies

- `os`: For directory and file operations.
- `requests`: For making HTTP requests.
- `bs4 (BeautifulSoup)`: For parsing HTML content.
- `datetime`: For handling date and time.
- `json`: For handling JSON data.
- `csv`: For handling CSV files.
- `argparse`: For parsing command-line arguments.

## Functions

### `load_names_from_csv(file_path)`

Load keywords to scrape from a CSV file and return the first keyword with status "Waiting".

**Parameters:**
- `file_path (str)`: The path to the CSV file containing the keywords and status.

**Returns:**
- `str`: The first keyword with status "Waiting".

### `update_status_in_csv(file_path, keyword, new_status)`

Update the status of a keyword in the CSV file.

**Parameters:**
- `file_path (str)`: The path to the CSV file containing the keywords and status.
- `keyword (str)`: The keyword to update.
- `new_status (str)`: The new status to set.

### `get_images_from_page(keyword, page_num, family='creative')`

Scrapes image URLs and alt texts from a Getty Images search results page.

**Parameters:**
- `keyword (str)`: The search keyword.
- `page_num (int)`: The page number to scrape.
- `family (str)`: The image family type, either 'creative' or 'editorial'.

**Returns:**
- `list`: A list of tuples containing image URLs and alt texts.

### `download_image(image_url, folder_name, keyword, image_id)`

Downloads an image from a given URL and saves it to a specified folder.

**Parameters:**
- `image_url (str)`: The URL of the image to download.
- `folder_name (str)`: The folder to save the image in.
- `keyword (str)`: The search keyword.
- `image_id (int)`: The ID of the image.

**Returns:**
- `str`: The name of the downloaded image file, or `None` if the download failed.

### `save_metadata_to_json(metadata, folder_name)`

Saves image metadata to a JSON file.

**Parameters:**
- `metadata (dict)`: The metadata to save.
- `folder_name (str)`: The folder to save the JSON file in.

### `scrape_images(keyword, max_pages=200, family='creative')`

Scrapes images based on a keyword and saves them along with their metadata.

**Parameters:**
- `keyword (str)`: The search keyword.
- `max_pages (int)`: The maximum number of pages to scrape.
- `family (str)`: The image family type, either 'creative' or 'editorial'.

## Command-Line Interface

The script can be run from the command line with the following arguments:

- `--keyword`: The keyword to search for images.
- `--max_pages`: The maximum number of pages to scrape (default is 200).
- `--csv_file`: The path to the CSV file containing keywords (default is "keywords.csv").

### Example Usage

```sh
python scrape.py --keyword "nature" --max_pages 10