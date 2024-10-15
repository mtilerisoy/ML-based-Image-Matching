import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
import csv
import argparse
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor

def load_names_from_csv(file_path):
    keywords = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['status'] == 'Waiting':
                keywords.append(row['keyword'])
    return keywords

def keyword_exists_in_csv(file_path, keyword):
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['keyword'] == keyword:
                return True
    return False

def update_status_in_csv(file_path, keyword, new_status):
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

async def download_image_async(session, image_url, folder_name, keyword, image_id):
    try:
        async with session.get(image_url) as response:
            if response.status == 200:
                timestamp = datetime.now().strftime("%d%m%Y")
                keyword_modified = keyword.replace(' ', '_')
                image_name = f"{keyword_modified}_{timestamp}_{image_id}.jpg"
                image_path = os.path.join(folder_name, image_name)
                
                with open(image_path, 'wb') as file:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        file.write(chunk)
                print(f"Downloaded: {image_name}")
                return image_name, image_url, None
            else:
                print(f"Failed to download: {image_url}")
                return None, image_url, "Failed to download"
    except Exception as e:
        print(f"Error downloading {image_url}: {str(e)}")
        return None, image_url, str(e)

def save_metadata_to_json(metadata, folder_name):
    json_file_path = os.path.join(folder_name, "metadata.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)
    print(f"Metadata saved to {json_file_path}")

async def scrape_images_async(keyword, max_pages=200, family='creative'):
    folder_name = keyword.replace(' ', '_')
    folder_name = os.path.join("..", "data", "scraped", "x_" + folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    image_id = 1
    metadata = {"images": []}
    consecutive_errors = 0

    async with aiohttp.ClientSession() as session:
        for page in range(1, max_pages + 1):
            print(f"Scraping page {page}")
            images_info = get_images_from_page(keyword, page, family=family)

            tasks = []
            for image_url, alt_text in images_info:
                tasks.append(download_image_async(session, image_url, folder_name, keyword, image_id))
                image_id += 1

            results = await asyncio.gather(*tasks)

            for image_name, image_url, error in results:
                if image_name:
                    metadata["images"].append({
                        "filename": image_name,
                        "caption": alt_text,
                        "match": False,
                        "design": "NONE",
                        "score": 0.0,
                        "URL": image_url
                    })
                    consecutive_errors = 0  # Reset the error counter on successful download
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= 10:
                        print("10 consecutive invalid URL errors encountered. Moving to the next keyword.")

            save_metadata_to_json(metadata, folder_name)
    
    new_folder_name = folder_name.replace("x_", "", 1)
    if os.path.exists(folder_name):
        os.rename(folder_name, new_folder_name)
        print(f"Renamed folder from {folder_name} to {new_folder_name}")
    else:
        print(f"Folder {folder_name} does not exist, cannot rename.")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Scrape images from Getty Images.")
    parser.add_argument("--keyword", type=str, help="The keyword to search for images.")
    parser.add_argument("--max_pages", type=int, default=200, help="The maximum number of pages to scrape.")
    parser.add_argument("--csv_file", type=str, default="keywords.csv", help="The path to the CSV file containing keywords.")
    args = parser.parse_args()

    if args.keyword:
        keywords = [args.keyword]
    else:
        keywords = load_names_from_csv(args.csv_file)
    
    if not keywords:
        print("No keywords provided and no keywords with status 'Waiting' found in the CSV file.")
    else:
        for keyword in keywords:
            keyword_in_csv = keyword_exists_in_csv(args.csv_file, keyword)
            if keyword_in_csv:
                update_status_in_csv(args.csv_file, keyword, "In Progress")
            
            success = asyncio.run(scrape_images_async(keyword, max_pages=args.max_pages, family='editorial'))
            
            if keyword_in_csv:
                new_status = "Completed" if success else "Failed"
                update_status_in_csv(args.csv_file, keyword, new_status)

if __name__ == "__main__":
    main()