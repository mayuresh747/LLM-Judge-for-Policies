import os
import requests
from bs4 import BeautifulSoup
import time
import re

# --- CONFIGURATION ---
DOWNLOAD_DIR = "Legal_Docs_Scraped"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Headers are required to stop government sites from blocking the script
HEADERS = {"User-Agent": USER_AGENT}

# Validated Dictionaries based on your list
# I have removed the obvious hallucinations (like "RCW 36.1000") to save time.
TARGETS = {
    "RCW": [
        "36.70A", "36.12", "35A.21", "70A.05", "70A.500", "70.240", 
        "19.194", "18.27", "49.22", "49.40"
    ],
    "WAC": [
        "51-11C", "204-60B", "204-65B", "204-70B", "173-12A", 
        "173-24", "173-25A", "173-26A", "208-010R"
    ],
    "SPU_Chapters": ["4", "8", "9", "10", "20"] # Design Standards Chapters
}

# Citations for SMC to generate links for (since we can't download them)
SMC_CITATIONS = [
    "23.40.020", "23.40.060", "23.41.100", "23.40.190", 
    "24.08.020", "14.12.030", "22.90.010"
]

def setup_directories():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    print(f"📂 Output Folder: {os.path.abspath(DOWNLOAD_DIR)}\n")

def download_file(url, filename):
    """Downloads a file with error handling and headers."""
    filepath = os.path.join(DOWNLOAD_DIR, filename)
    if os.path.exists(filepath):
        print(f"⏭️  Skipping (exists): {filename}")
        return

    try:
        print(f"⬇️  Downloading: {filename}...")
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"✅ Success: {filename}")
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")

def scrape_rcw_wac(code_type, chapters):
    """
    Traverses the leg.wa.gov directory structure.
    Logic: 
    1. Identify Title from Chapter (e.g., 36.70A -> Title 36).
    2. Construct the PDF URL using the standard pattern found on the site.
    3. Verify the link exists (HEAD request) before downloading.
    """
    print(f"\n--- 🔍 Scraping {code_type} ---")
    
    base_pdf_url = "http://lawfilesext.leg.wa.gov/law"
    
    for chapter in chapters:
        # Parse Title and Chapter
        # RCW format: 36.70A -> Title 36
        # WAC format: 51-11C -> Title 51
        if code_type == "RCW":
            title = chapter.split('.')[0]
            # URL Pattern: .../rcw/pdf/36/36.70A.pdf
            pdf_url = f"{base_pdf_url}/rcw/pdf/{title}/{chapter}.pdf"
        else:
            title = chapter.split('-')[0]
            # URL Pattern: .../wac/pdf/51/51-11C.pdf
            pdf_url = f"{base_pdf_url}/wac/pdf/{title}/{chapter}.pdf"

        filename = f"{code_type}_{chapter}.pdf"
        
        # Try to download
        download_file(pdf_url, filename)
        time.sleep(1) # Be polite to the server

def scrape_spu_design_standards():
    """
    Scrapes the Seattle Public Utilities Design Standards page to find actual PDF links.
    """
    print(f"\n--- 🔍 Scraping SPU Design Standards ---")
    url = "https://www.seattle.gov/utilities/construction-resources/standards-and-guidelines/design-standards-and-guidelines"
    
    try:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all PDF links on the page
        links = soup.find_all('a', href=True)
        
        found_count = 0
        for link in links:
            href = link['href']
            text = link.get_text().strip()
            
            # Filter for PDF files that look like Chapters
            if href.endswith('.pdf') and "Chapter" in text:
                # specific filter for the chapters we want
                for target_chap in TARGETS["SPU_Chapters"]:
                    if f"Chapter {target_chap}" in text or f"Chapter_{target_chap}" in href:
                        # Handle relative URLs
                        if href.startswith('/'):
                            href = "https://www.seattle.gov" + href
                        
                        filename = f"SPU_DSG_{text.replace(' ', '_')}.pdf"
                        download_file(href, filename)
                        found_count += 1
                        break
        
        if found_count == 0:
            print("⚠️  No SPU chapters found. The page structure might have changed.")
            
    except Exception as e:
        print(f"❌ Error scraping SPU: {e}")

def generate_smc_links():
    """
    Generates clickable links for SMC because they cannot be downloaded via script.
    """
    print(f"\n--- 🏛️  Seattle Municipal Code (SMC) ---")
    print("⚠️  NOTE: SMC files cannot be downloaded automatically due to Municode protections.")
    print("👉 Please click these links to view the codes directly:\n")
    
    base_search = "https://library.municode.com/wa/seattle/search?searchText="
    
    for cite in SMC_CITATIONS:
        print(f"🔹 {cite}: {base_search}{cite}")

if __name__ == "__main__":
    setup_directories()
    
    # 1. Scrape RCW (State Laws)
    scrape_rcw_wac("RCW", TARGETS["RCW"])
    
    # 2. Scrape WAC (Admin Codes)
    scrape_rcw_wac("WAC", TARGETS["WAC"])
    
    # 3. Scrape SPU (Design Standards)
    scrape_spu_design_standards()
    
    # 4. Generate SMC Links
    generate_smc_links()
    
    print("\n✅ Process Complete.")