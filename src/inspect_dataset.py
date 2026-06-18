import os
from PIL import Image

def analyze_dataset(data_dir="data/raw"):
    if not os.path.exists(data_dir):
        print(f"❌ Error: Directory '{data_dir}' does not exist.")
        return

    print("🔍 Analyzing downloaded dataset...")

    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if len(subdirs) == 1 and subdirs[0].lower() in ['images', 'cat-breeds-dataset', 'dataset']:
        data_dir = os.path.join(data_dir, subdirs[0])
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    total_images = 0
    corrupted_files = []
    breed_counts = {}

    for breed in subdirs:
        breed_path = os.path.join(data_dir, breed)
        files = os.listdir(breed_path)
        valid_images_in_breed = 0
        
        for file in files:
            file_path = os.path.join(breed_path, file)
            if file.startswith('.') or not os.path.isfile(file_path):
                continue
                
            try:
                with Image.open(file_path) as img:
                    img.verify()
                valid_images_in_breed += 1
                total_images += 1
            except Exception:
                corrupted_files.append(file_path)

        breed_counts[breed] = valid_images_in_breed

    print("\n" + "="*40)
    print("        DATASET METRICS SUMMARY        ")
    print("="*40)
    print(f"Total Unique Breeds Found: {len(subdirs)}")
    print(f"Total Verifiable Images:   {total_images}")
    print(f"Total Corrupted Files:     {len(corrupted_files)}")
    print("="*40)

    print("\n📊 Top 5 Most Populated Breeds:")
    sorted_breeds = sorted(breed_counts.items(), key=lambda x: x[1], reverse=True)
    for breed, count in sorted_breeds[:5]:
        print(f"  - {breed}: {count} images")

    print("\n📊 Bottom 5 Least Populated Breeds (Class Imbalance Risk):")
    for breed, count in sorted_breeds[-5:]:
        print(f"  - {breed}: {count} images")

    if corrupted_files:
        print(f"\n⚠️ Found {len(corrupted_files)} unreadable files. We will automatically strip these out during preprocessing.")

if __name__ == "__main__":
    analyze_dataset()