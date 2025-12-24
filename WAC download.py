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
    safe_text = re.sub(r'[\\/*?:"<>|]', "", text)
    safe_text = re.sub(r'\s+', '_', safe_text)
    return safe_text.strip('_')

def download_file(url, folder, filename):
    """Helper to download a file with Content-Type checking."""
    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        print(f"  - File already exists: {filename}")
        return

    try:
        print(f"  - Downloading {filename}...")
        # Stream the request to check headers before downloading body
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        # VALIDATION: Check if the server is sending us a PDF or HTML
        content_type = response.headers.get('Content-Type', '').lower()
        if 'html' in content_type:
            print(f"    ERROR: Server returned HTML instead of PDF. URL might be wrong: {url}")
            return
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("    Success!")
    except Exception as e:
        print(f"    Failed to download: {e}")

def main():
    print("Starting WAC Scraper...")
    
    # 1. Get List of Titles
    main_soup = get_soup(BASE_URL)
    if not main_soup:
        return

    title_links = []
    for a in main_soup.find_all('a', href=True):
        if 'cite=' in a['href'] and 'Title' in a.get_text():
            full_link = urljoin(BASE_URL, a['href'])
            title_text = a.get_text().strip()
            folder_name = clean_filename(title_text.split('|')[0] if '|' in title_text else title_text)
            title_links.append((folder_name, full_link))

    print(f"Found {len(title_links)} Titles.")

    # 2. Iterate Titles
    for title_folder_name, title_url in title_links:
        print(f"\nProcessing {title_folder_name}...")
        
        title_dir = os.path.join(OUTPUT_DIR, title_folder_name)
        if not os.path.exists(title_dir):
            os.makedirs(title_dir)

        title_soup = get_soup(title_url)
        if not title_soup:
            continue

        # 3. Get Chapters
        chapter_data = []
        # Try table row logic first (most reliable for descriptions)
        for tr in title_soup.find_all('tr'):
            link_tag = tr.find('a', href=True)
            if link_tag and 'cite=' in link_tag['href']:
                chapter_num = link_tag.get_text().strip()
                if chapter_num and chapter_num[0].isdigit() and '-' in chapter_num:
                    full_chapter_url = urljoin(BASE_URL, link_tag['href'])
                    full_row_text = tr.get_text(" ", strip=True)
                    chapter_name = full_row_text if full_row_text else chapter_num
                    chapter_data.append((chapter_name, full_chapter_url))
        
        # Fallback if table logic fails
        if not chapter_data:
             for a in title_soup.find_all('a', href=True):
                if 'cite=' in a['href']:
                    txt = a.get_text().strip()
                    if txt and txt[0].isdigit() and '-' in txt:
                        desc = a.next_sibling
                        full_name = txt + " " + (desc.strip() if desc else "")
                        chapter_data.append((full_name, urljoin(BASE_URL, a['href'])))

        print(f"  Found {len(chapter_data)} Chapters in {title_folder_name}.")

        # 4. Iterate Chapters
        for raw_chapter_name, chapter_url in chapter_data:
            clean_name = clean_filename(raw_chapter_name)
            if len(clean_name) > 100: clean_name = clean_name[:100]
            filename = f"WAC_{clean_name}.pdf"
            
            if os.path.exists(os.path.join(title_dir, filename)):
                print(f"  - Skipping {filename}")
                continue

            chapter_soup = get_soup(chapter_url)
            if not chapter_soup:
                continue

            # 5. FIND THE PDF LINK (FIXED LOGIC)
            pdf_link = None
            
            # Strategy A: Look for explicit "Complete Chapter" header/container
            # The site often has "Complete Chapter" text followed by a PDF link
            header = chapter_soup.find(string=re.compile("Complete Chapter", re.IGNORECASE))
            if header:
                # Look in the parent container for a link with "PDF" text or .pdf extension
                parent = header.find_parent()
                if parent:
                    # Try to find a link with text "PDF"
                    target_link = parent.find_next('a', string=re.compile("PDF", re.IGNORECASE))
                    if not target_link:
                         # Or any link ending in .pdf nearby
                         target_link = parent.find_next('a', href=re.compile(r'\.pdf$', re.IGNORECASE))
                    
                    if target_link:
                        pdf_link = urljoin(BASE_URL, target_link['href'])

            # Strategy B: If A failed, look globally for a link ending in .pdf
            # (ignoring Section PDFs which usually don't appear until further down)
            if not pdf_link:
                for a in chapter_soup.find_all('a', href=True):
                    # Check for direct PDF file extension
                    if a['href'].lower().endswith('.pdf'):
                        pdf_link = urljoin(BASE_URL, a['href'])
                        # Usually the first PDF on the page is the complete chapter
                        break

            if pdf_link:
                download_file(pdf_link, title_dir, filename)
            else:
                print(f"  - WARNING: Could not find 'Complete Chapter' PDF for {clean_name}")

            time.sleep(1)

if __name__ == "__main__":
    main()