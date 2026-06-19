import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

IMG_SIZE = (224, 224)

def load_class_names(data_dir="data/processed"):
    """
    Keras image_dataset_from_directory sorts classes alphanumerically.
    We read the folder names to reconstruct the label map.
    """
    classes = sorted(os.listdir(data_dir))
    return classes

def predict_cat_breed(img_path, model_path="models/mobilenet_cat_identifier.keras"):
    print(f"🔍 Loading AI Model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    class_names = load_class_names()
    
    print(f"📸 Processing Image: {img_path}")
    # 1. Load and resize the image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    
    # 2. Convert to a numpy array
    img_array = image.img_to_array(img)
    
    # 3. Scale pixels to [0, 1] exactly like our training pipeline did
    img_array = img_array / 255.0
    
    # 4. Expand dimensions (the model expects a batch, so we make it a batch of 1)
    img_array = tf.expand_dims(img_array, 0)
    
    # 5. Make the prediction
    predictions = model.predict(img_array)
    
    # 6. Get the top 3 predictions
    # argsort sorts ascending, so we take the last 3 and reverse them
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    
    print("\n" + "="*40)
    print("         AI PREDICTION RESULTS         ")
    print("="*40)
    for i, index in enumerate(top_3_indices):
        breed = class_names[index]
        confidence = predictions[0][index] * 100
        print(f"  {i+1}. {breed}: {confidence:.2f}% confidence")
    print("="*40)

if __name__ == "__main__":
    # Let's test it on a random image from the validation set
    # Feel free to change this path to a picture of your own cat!
    
    # Safely find the first folder and first image to use as a test
    data_dir = "data/processed"
    first_breed = sorted(os.listdir(data_dir))[0]
    first_image = os.listdir(os.path.join(data_dir, first_breed))[0]
    
    test_image_path = os.path.join(data_dir, first_breed, first_image)
    
    predict_cat_breed(test_image_path)