import os
from PIL import Image

def analyze_dataset(root_dir="data/raw"):
    print("🔍 Analyzing downloaded dataset architecture...")
    
    total_images = 0
    corrupted_files = []
    breed_counts = {}
    
    for subdir, dirs, files in os.walk(root_dir):
        valid_images_in_breed = 0
        breed_name = os.path.basename(subdir)
        
        for file in files:
            file_path = os.path.join(subdir, file)

            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
                continue
                
            try:
                with Image.open(file_path) as img:
                    img.verify()
                valid_images_in_breed += 1
                total_images += 1
            except Exception:
                corrupted_files.append(file_path)
                
        if valid_images_in_breed > 0:
            breed_counts[breed_name] = valid_images_in_breed

    print("\n" + "="*40)
    print("        DATASET METRICS SUMMARY        ")
    print("="*40)
    print(f"Total Unique Breeds Found: {len(breed_counts)}")
    print(f"Total Verifiable Images:   {total_images}")
    print(f"Total Corrupted Files:     {len(corrupted_files)}")
    print("="*40)

    if not breed_counts:
        print("❌ No images found! The dataset structure might be heavily corrupted.")
        return

    print("\n📊 Top 5 Most Populated Breeds:")
    sorted_breeds = sorted(breed_counts.items(), key=lambda x: x[1], reverse=True)
    for breed, count in sorted_breeds[:5]:
        print(f"  - {breed}: {count} images")

    print("\n📊 Bottom 5 Least Populated Breeds (Class Imbalance Risk):")
    for breed, count in sorted_breeds[-5:]:
        print(f"  - {breed}: {count} images")

    if corrupted_files:
        print(f"\n⚠️ Found {len(corrupted_files)} unreadable files. These should be removed during preprocessing.")

if __name__ == "__main__":
    analyze_dataset()