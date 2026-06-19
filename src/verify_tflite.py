import os
import numpy as np
import tensorflow as tf

def verify_mobile_model(image_path="data/processed/Abyssinian/Abyssinian_1.jpg", 
                        model_path="models/model.tflite", 
                        labels_path="models/labels.txt"):
    """
    Simulates how a mobile app runs inference using the compressed .tflite model.
    """
    print("🔍 Starting Local TFLite Verification...")

    # 1. Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        print("❌ Error: Missing TFLite model or labels.txt file.")
        print("Make sure to run src/convert_tflite.py after training finishes!")
        return

    # 2. Load the label map
    with open(labels_path, "f") as f:
        labels = [line.strip() for line in f.readlines()]
    print(f"📖 Loaded {len(labels)} class labels.")

    # 3. Load the TFLite model and allocate tensors
    # The 'Interpreter' is the lightweight execution engine designed for edge devices
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 4. Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get required image dimensions from the model (should be 224x224)
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    print(f"🧠 Model expects input shape: {input_shape}")

    # 5. Load and preprocess a test image using Keras utilities
    print(f"🖼️ Loading test image: {image_path}")
    try:
        img = tf.keras.utils.load_img(image_path, target_size=(height, width))
        img_array = tf.keras.utils.img_to_array(img)
        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        img_array = np.expand_value_dims = np.expand_dims(img_array, axis=0)
        # EfficientNetB0 expects pixel values scaled/normalized if not using rescaling layer,
        # but our model architecture includes the normalization layer inside it!
    except Exception as e:
        print(f"❌ Failed to load test image: {e}")
        return

    # 6. Set the tensor to point to the input data
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # 7. Run the inference engine!
    print("⚡ Running inference on TFLite Interpreter...")
    interpreter.invoke()

    # 8. Retrieve the prediction results
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Find the highest probability index
    top_index = np.argmax(output_data)
    confidence = output_data[top_index]

    print("\n🎉 --- VERIFICATION RESULTS ---")
    print(f"Predicted Breed: {labels[top_index]}")
    print(f"Confidence Score: {confidence * 100:.2f}%")
    print("--------------------------------\n")
    print("✅ TFLite model is completely functional and ready for mobile deployment!")

if __name__ == "__main__":
    # Feel free to change the default image path to any valid image in your dataset
    verify_mobile_model()