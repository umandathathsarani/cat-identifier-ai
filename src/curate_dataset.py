import os
import shutil
import random

def curate_dataset(raw_dir="data/raw", processed_dir="data/processed", min_images=150, max_images=1000):
    print("🧹 Starting dataset curation...")
    print(f"Rules: Minimum {min_images} images, Maximum {max_images} images per breed.\n")

    # Create processed directory
    os.makedirs(processed_dir, exist_ok=True)

    breed_data = {}

    # 1. Collect all valid images per breed
    for subdir, dirs, files in os.walk(raw_dir):
        breed_name = os.path.basename(subdir)
        valid_images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        if len(valid_images) > 0:
            if breed_name not in breed_data:
                breed_data[breed_name] = []
            
            for img in valid_images:
                breed_data[breed_name].append(os.path.join(subdir, img))

    # 2. Filter, Sample, and Copy
    approved_breeds = 0
    copied_images = 0

    for breed, images in breed_data.items():
        count = len(images)
        
        # Rule 1: The Floor (Drop underrepresented classes)
        if count < min_images:
            print(f"  [SKIPPED] {breed}: Only {count} images (Below minimum)")
            continue
            
        approved_breeds += 1
        
        # Rule 2: The Ceiling (Cap overrepresented classes)
        random.shuffle(images)
        selected_images = images[:max_images]
        
        dest_folder = os.path.join(processed_dir, breed)
        os.makedirs(dest_folder, exist_ok=True)
        
        # Copy files to the processed folder
        for img_path in selected_images:
            filename = os.path.basename(img_path)
            dest_path = os.path.join(dest_folder, filename)
            shutil.copy2(img_path, dest_path)
            copied_images += 1
            
        print(f"  [COPIED] {breed}: Processed {len(selected_images)} images.")

    # Print Summary Report
    print("\n" + "="*40)
    print("        CURATION COMPLETE        ")
    print("="*40)
    print(f"Final Approved Breeds:    {approved_breeds}")
    print(f"Total Processed Images:   {copied_images}")
    print(f"Ready for AI Training in: {processed_dir}")
    print("="*40)

if __name__ == "__main__":
    # Seed the random number generator so you get the exact same images if you run this twice
    random.seed(42)
    curate_dataset()