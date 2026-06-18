import os
from bing_image_downloader import downloader

def build_dataset(breeds, num_images=50, output_dir="data/raw"):
    """
    Scrapes images from Bing and categorizes them into folders.
    """
    print(f"Initializing dataset download to: {output_dir}\n")

    os.makedirs(output_dir, exist_ok=True)

    for breed in breeds:
        print(f"\n--- Fetching images for: {breed} ---")
        query = f"{breed} cat" 
        
        try:
            downloader.download(
                query,
                limit=num_images,
                output_dir=output_dir,
                adult_filter_off=True, 
                force_replace=False, 
                timeout=60,
                verbose=False 
            )
        except Exception as e:
            print(f"Failed to download {breed}: {e}")

if __name__ == "__main__":
    test_breeds = [
        "Abyssinian",
        "Bengal",
        "Sphynx"
    ]

    build_dataset(test_breeds, num_images=20)
    
    print("\n✅ Test download complete! Check your data/raw/ folder.")