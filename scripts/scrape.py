import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
import csv

def load_names_from_csv(file_path):
    """
    Load keywords from a CSV file and return the first keyword with status "Waiting".

    Parameters:
    file_path (str): The path to the CSV file containing the keywords and status.

    Returns:
    str: The first keyword with status "Waiting".
    """
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['status'] == 'Waiting':
                return row['keyword']
    return None

def update_status_in_csv(file_path, keyword, new_status):
    """
    Update the status of a keyword in the CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing the keywords and status.
    keyword (str): The keyword to update.
    new_status (str): The new status to set.
    """
    rows = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['keyword'] == keyword:
                row['status'] = new_status
            rows.append(row)

    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['keyword', 'status'])
        writer.writeheader()
        writer.writerows(rows)

HEADERS = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-GB,en;q=0.9",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15"
}

def get_images_from_page(keyword, page_num, family='creative'):
    assert family in ['creative', 'editorial'], "Family must be either 'creative' or 'editorial'"
    keyword = keyword.replace(' ', '%20')
    url = f"https://www.gettyimages.nl/search/2/image?family={family}&page={page_num}&phrase={keyword}&sort=best"
    print(f"Scraping: {url}")
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')

    images_info = []
    img_tags = soup.find_all('img')

    for img_tag in img_tags:
        image_url = img_tag.get('src')
        alt_text = img_tag.get('alt')
        if image_url and alt_text:
            images_info.append((image_url, alt_text))

    return images_info

def download_image(image_url, folder_name, keyword, image_id):
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            timestamp = datetime.now().strftime("%d%m%Y")
            keyword_modified = keyword.replace(' ', '_')
            image_name = f"{keyword_modified}_{timestamp}_{image_id}.jpg"
            image_path = os.path.join(folder_name, image_name)
            
            with open(image_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Downloaded: {image_name}")
            return image_name
        else:
            print(f"Failed to download: {image_url}")
            return None
    except Exception as e:
        print(f"Error downloading {image_url}: {str(e)}")
        return None

def save_metadata_to_json(metadata, folder_name):
    json_file_path = os.path.join(folder_name, "metadata.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)
    print(f"Metadata saved to {json_file_path}")

def scrape_images(keyword, max_pages=5, family='creative'):
    folder_name = keyword.replace(' ', '_')
    folder_name = "x_" + folder_name
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    image_id = 1
    metadata = {"images": []}
    consecutive_errors = 0

    for page in range(1, max_pages + 1):
        print(f"Scraping page {page}")
        images_info = get_images_from_page(keyword, page, family=family)

        for image_url, alt_text in images_info:
            image_name = download_image(image_url, folder_name, keyword, image_id)
            if image_name:
                metadata["images"].append({
                    "filename": image_name,
                    "caption": alt_text,
                    "match": False,
                    "design": "NONE",
                    "score": 0.0,
                    "URL": image_url
                })
                image_id += 1
                consecutive_errors = 0  # Reset the error counter on successful download
            else:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    print("5 consecutive invalid URL errors encountered. Stopping the process.")
                    return

            save_metadata_to_json(metadata, folder_name)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, "keywords.csv")
    keyword = load_names_from_csv(csv_file_path)
    if keyword:
        # update_status_in_csv(csv_file_path, keyword, "In Progress")
        scrape_images(keyword, max_pages=200, family='editorial')
        update_status_in_csv(csv_file_path, keyword, "Completed")
    else:
        print("No keywords with status 'Waiting' found.")