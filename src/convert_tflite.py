import os
import tensorflow as tf

def export_labels(data_dir="data/processed", output_path="models/labels.txt"):
    """
    Extracts the class names from the directory structure
    and saves them to a text file for the mobile app to read.
    """
    print("📝 Generating label map for mobile app...")
    class_names = sorted(os.listdir(data_dir))
    
    with open(output_path, "w") as f:
        for breed in class_names:
            f.write(f"{breed}\n")
            
    print(f"✅ Saved {len(class_names)} labels to {output_path}")

def convert_to_tflite(model_path="models/efficientnet_cat_identifier.keras", output_path="models/model.tflite"):
    """
    Converts a standard Keras model into a highly compressed, 
    mobile-optimized TensorFlow Lite model.
    """
    print(f"⚙️ Loading trained model from {model_path}...")
    
    # 1. Load the heavy desktop model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure your training script has completely finished running first!")
        return

    # 2. Initialize the converter
    print("🚀 Initializing TFLite Converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 3. Apply Optimizations
    # This specifically tells TensorFlow to compress the model weights (e.g., converting 32-bit floats to 16-bit or 8-bit integers). This drastically reduces the file size with almost zero loss in accuracy.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 4. Convert the model
    print("⏳ Compressing and converting model (this might take a minute)...")
    tflite_model = converter.convert()
    
    # 5. Save the mobile-ready file
    with open(output_path, "wb") as f:
        f.write(tflite_model)
        
    # Calculate the file size in Megabytes
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Conversion complete! TFLite model saved to {output_path} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Run the extraction and conversion
    export_labels()
    convert_to_tflite()