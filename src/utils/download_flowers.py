import os
import urllib.request
import tarfile
import shutil

DATA_DIR = "data/flowers102"
URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
TGZ_FILE = "102flowers.tgz"

def download_monitor(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = (downloaded / total_size) * 100
    print(f"\rDownloading: {percent:.1f}% ({downloaded//1024//1024}MB / {total_size//1024//1024}MB)", end="")

def main():
    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 100:
        print(f"Directory {DATA_DIR} already exists and is not empty. Skipping.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download
    print(f"Downloading Flowers102 from {URL}...")
    try:
        urllib.request.urlretrieve(URL, TGZ_FILE, download_monitor)
        print("\nDownload complete. Extracting...")
    except Exception as e:
        print(f"\nError downloading: {e}")
        return

    # Extract
    try:
        with tarfile.open(TGZ_FILE, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
        print("Extraction complete.")
        
        # The tar contains a 'jpg' folder. Let's move files up if needed, or just leave them.
        # Structure is usually data/flowers102/jpg/*.jpg
        # Let's verify structure
        extracted_jpg_dir = os.path.join(DATA_DIR, "jpg")
        if os.path.exists(extracted_jpg_dir):
            print(f"Images are in {extracted_jpg_dir}")
            # Optional: Move them to root of data/flowers102 for simplicity, but subdirectory is fine for ImageFolder/Inference
    except Exception as e:
        print(f"Error extracting: {e}")
    finally:
        # Cleanup
        if os.path.exists(TGZ_FILE):
            os.remove(TGZ_FILE)
            print("Cleaned up tar file.")

if __name__ == "__main__":
    main()
