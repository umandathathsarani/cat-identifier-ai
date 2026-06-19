import os
import tensorflow as tf
from preprocess import create_data_pipeline

def build_model(num_classes):
    print("🏗️ Building MobileNetV3 Transfer Learning Architecture...")

    # 1. Load the pre-trained base model
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False 

    # 2. Build the custom classification head
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    x = inputs * 255.0  
    
    # Pass inputs through the frozen base model
    x = base_model(x, training=False)
    
    # Condense the massive 2D feature maps into a 1D array
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Add Dropout to prevent overfitting (randomly turns off 20% of neurons during training)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Final Output Layer: One neuron for each of your 42 cat breeds
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # 3. Compile the full model
    model = tf.keras.Model(inputs, outputs)
    
    # Use sparse_categorical_crossentropy because our labels are integers, not one-hot encoded arrays
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def start_training():
    os.makedirs("models", exist_ok=True)

    # 1. Load the data
    train_ds, val_ds, class_names = create_data_pipeline()
    num_classes = len(class_names)
    
    # 2. Build the model
    model = build_model(num_classes)
    model.summary()

    # 3. Setup Callbacks
    callbacks = [     
        # ModelCheckpoint: Saves the model ONLY when validation accuracy improves
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/mobilenet_cat_identifier.keras",
            save_best_only=True,
            monitor="val_accuracy"
        ),
        # EarlyStopping: Stops training if the model stops learning for 3 epochs to save time
        tf.keras.callbacks.EarlyStopping(
            patience=3, 
            restore_best_weights=True,
            monitor="val_loss"
        )
    ]

    print("\n🚀 Commencing Initial Training Phase...")
    # 4. Train the model (Starting with just 5 epochs for a test run)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=callbacks
    )
    
    print("\n✅ Training Complete. Best model saved to 'models/mobilenet_cat_identifier.keras'")

if __name__ == "__main__":
    start_training()