import os
import tensorflow as tf
import numpy as np
from preprocess import create_data_pipeline

def calculate_class_weights(data_dir="data/processed"):
    """
    Calculates weights to mathematically force the AI to care equally 
    about a breed with 150 images and a breed with 1000 images.
    Formula: total_samples / (num_classes * class_samples)
    """
    class_counts = {}
    for idx, breed in enumerate(sorted(os.listdir(data_dir))):
        class_counts[idx] = len(os.listdir(os.path.join(data_dir, breed)))
        
    total_images = sum(class_counts.values())
    num_classes = len(class_counts)
    
    class_weights = {
        idx: total_images / (num_classes * count) 
        for idx, count in class_counts.items()
    }
    print(f"⚖️ Class weights calculated to balance {num_classes} classes.")
    return class_weights

def build_advanced_model(num_classes):
    print("🏗️ Building EfficientNetB0 Architecture...")

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False 

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = inputs * 255.0  
    x = base_model(x, training=False)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x) 
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def start_training():
    os.makedirs("models", exist_ok=True)
    train_ds, val_ds, class_names = create_data_pipeline()
    num_classes = len(class_names)
    
    weights = calculate_class_weights()
    
    model, base_model = build_advanced_model(num_classes)

    print("\n🚀 PHASE 1: Training Custom Head (15 Epochs)...")
    
    phase1_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy")
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15, 
        class_weight=weights,
        callbacks=phase1_callbacks
    )

    print("\n🔬 PHASE 2: Unfreezing top 50 layers of EfficientNet...")
    
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    phase2_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/efficientnet_cat_identifier.keras",
            save_best_only=True,
            monitor="val_accuracy"
        ),

        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_accuracy")
    ]

    print("\n🚀 Commencing Deep Fine-Tuning (Up to 40 Epochs)...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=40, 
        class_weight=weights,
        callbacks=phase2_callbacks
    )
    
    print("\n✅ Advanced Training Complete. Best model saved to 'models/efficientnet_cat_identifier.keras'")

if __name__ == "__main__":
    start_training()