import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote, urlparse
import time

def download_wac_pdfs_clean():
    url = "https://app.leg.wa.gov/WAC/default.aspx?cite=51-11C"
    save_dir = "wac_pdfs"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    print(f"Fetching page: {url}")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching page: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    # Use a SET to store URLs. A set automatically removes duplicates.
    unique_pdf_urls = set()

    # Find all links on the page
    all_links = soup.find_all('a', href=True)

    for link in all_links:
        href = link['href']
        
        # Filter: Must contain '.pdf'
        if '.pdf' in href.lower():
            # Convert relative link (e.g. "../files/x.pdf") to absolute (https://...)
            full_url = urljoin(url, href)
            unique_pdf_urls.add(full_url)

    # Convert back to a list and sort so they download in order
    sorted_urls = sorted(list(unique_pdf_urls))

    print(f"Found {len(sorted_urls)} unique PDF files.")

    if len(sorted_urls) == 0:
        print("No PDFs found. The website might block scripts or use JavaScript to load files.")
        return

    # Download Loop
    for i, full_url in enumerate(sorted_urls):
        try:
            # Get the original filename from the URL (e.g., "51-11C-10000.pdf")
            parsed_url = urlparse(full_url)
            filename = os.path.basename(unquote(parsed_url.path))

            file_path = os.path.join(save_dir, filename)

            # Optional: Check if file already exists to save time
            if os.path.exists(file_path):
                print(f"[{i+1}/{len(sorted_urls)}] Skipping {filename} (Exists)")
                continue

            print(f"[{i+1}/{len(sorted_urls)}] Downloading {filename}...")

            # Download
            with requests.get(full_url, headers=headers, stream=True) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Short pause to be polite
            time.sleep(0.3)

        except Exception as e:
            print(f"Failed to download {full_url}: {e}")

    print("\nDownload complete.")

if __name__ == "__main__":
    download_wac_pdfs_clean()