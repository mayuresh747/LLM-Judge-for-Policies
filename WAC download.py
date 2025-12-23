import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin
import re

# Base URLs
BASE_URL = "https://app.leg.wa.gov/WAC/default.aspx"

# Output directory
OUTPUT_DIR = "All Documents/WAC_Chapters"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_soup(url):
    """Helper to fetch a page and return a BeautifulSoup object."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def clean_filename(text):
    """Sanitize string to be valid filename."""
    # Replace invalid characters with underscore
    safe_text = re.sub(r'[\\/*?:"<>|]', "", text)
    # Replace multiple spaces/newlines with single underscore
    safe_text = re.sub(r'\s+', '_', safe_text)
    return safe_text.strip('_')

def download_file(url, folder, filename):
    """Helper to download a file."""
    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        print(f"  - File already exists: {filename}")
        return

    try:
        print(f"  - Downloading {filename}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("    Success!")
    except Exception as e:
        print(f"    Failed to download: {e}")

def main():
    print("Starting WAC Scraper...")
    
    # 1. Get List of Titles from the Main Page
    main_soup = get_soup(BASE_URL)
    if not main_soup:
        return

    title_links = []
    # Identify Title links
    for a in main_soup.find_all('a', href=True):
        if 'cite=' in a['href'] and 'Title' in a.get_text():
            full_link = urljoin(BASE_URL, a['href'])
            title_text = a.get_text().strip()
            # Create a simple folder name for the Title (e.g., "Title_1")
            folder_name = clean_filename(title_text.split('|')[0] if '|' in title_text else title_text)
            title_links.append((folder_name, full_link))

    print(f"Found {len(title_links)} Titles.")

    # 2. Iterate through each Title
    for title_folder_name, title_url in title_links:
        print(f"\nProcessing {title_folder_name}...")
        
        # Create subfolder for Title
        title_dir = os.path.join(OUTPUT_DIR, title_folder_name)
        if not os.path.exists(title_dir):
            os.makedirs(title_dir)

        title_soup = get_soup(title_url)
        if not title_soup:
            continue

        # 3. Get List of Chapters from the Title Page
        chapter_data = []
        
        # Iterate over all rows (tr) to find chapters and their descriptions
        # WAC lists are often in tables; this grabs the text from the whole row
        for tr in title_soup.find_all('tr'):
            link_tag = tr.find('a', href=True)
            if link_tag and 'cite=' in link_tag['href']:
                chapter_num = link_tag.get_text().strip()
                
                # Check if it looks like a chapter (e.g., "1-06")
                if chapter_num and chapter_num[0].isdigit() and '-' in chapter_num:
                    full_chapter_url = urljoin(BASE_URL, link_tag['href'])
                    
                    # Extract the full text of the row to get the description
                    # Example row text: "1-06 Public records."
                    full_row_text = tr.get_text(" ", strip=True)
                    
                    # If we found text, use it; otherwise fallback to just number
                    chapter_name = full_row_text if full_row_text else chapter_num
                    
                    chapter_data.append((chapter_name, full_chapter_url))

        # Fallback: If table logic failed (site layout differences), try standard links
        if not chapter_data:
             for a in title_soup.find_all('a', href=True):
                if 'cite=' in a['href']:
                    txt = a.get_text().strip()
                    if txt and txt[0].isdigit() and '-' in txt:
                        # Try to grab sibling text if not in a table
                        desc = a.next_sibling
                        full_name = txt + " " + (desc.strip() if desc else "")
                        chapter_data.append((full_name, urljoin(BASE_URL, a['href'])))

        print(f"  Found {len(chapter_data)} Chapters in {title_folder_name}.")

        # 4. Iterate through each Chapter
        for raw_chapter_name, chapter_url in chapter_data:
            # Create clean filename: "WAC_1-06_Public_records.pdf"
            clean_name = clean_filename(raw_chapter_name)
            # Ensure it's not too long (filesystem limits)
            if len(clean_name) > 100:
                clean_name = clean_name[:100]
            
            filename = f"WAC_{clean_name}.pdf"
            
            # Check if file exists
            if os.path.exists(os.path.join(title_dir, filename)):
                print(f"  - Skipping {filename} (already exists)")
                continue

            chapter_soup = get_soup(chapter_url)
            if not chapter_soup:
                continue

            # 5. Find the "Complete Chapter" PDF link
            pdf_link = None
            # Search for the specific "Complete Chapter" PDF icon/link
            for a in chapter_soup.find_all('a', href=True):
                # Criteria: has 'pdf' in text or href, and 'full=true' indicating complete chapter
                if 'full=true' in a['href']:
                    pdf_link = urljoin(BASE_URL, a['href'])
                    break
            
            # Fallback search strategies
            if not pdf_link:
                # Look for "Complete Chapter" text, then find the PDF link nearby
                header_node = chapter_soup.find(string=re.compile("Complete Chapter", re.IGNORECASE))
                if header_node:
                    parent_container = header_node.find_parent()
                    if parent_container:
                        possible_link = parent_container.find_next('a', href=re.compile(r'\.pdf', re.IGNORECASE))
                        if possible_link:
                            pdf_link = urljoin(BASE_URL, possible_link['href'])

            if pdf_link:
                download_file(pdf_link, title_dir, filename)
            else:
                print(f"  - WARNING: Could not find 'Complete Chapter' PDF for {clean_name}")

            time.sleep(1)

if __name__ == "__main__":
    main()