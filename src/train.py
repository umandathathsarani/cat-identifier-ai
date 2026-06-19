import os
import tensorflow as tf
from preprocess import create_data_pipeline

def build_model(num_classes):
    print("🏗️ Building MobileNetV3 Transfer Learning Architecture...")

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False 

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = inputs * 255.0  
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x) 
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
    
    model, base_model = build_model(num_classes)

    print("\n🚀 PHASE 1: Training the custom classification head...")
    
    phase1_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=3, 
            restore_best_weights=True,
            monitor="val_accuracy"
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10, 
        callbacks=phase1_callbacks
    )

    print("\n🔬 PHASE 2: Unfreezing top layers for fine-tuning...")
    
    base_model.trainable = True
    
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    phase2_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/mobilenet_cat_identifier.keras",
            save_best_only=True,
            monitor="val_accuracy"
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=4, 
            restore_best_weights=True,
            monitor="val_accuracy"
        )
    ]

    print("\n🚀 Commencing Fine-Tuning...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15, 
        callbacks=phase2_callbacks
    )
    
    print("\n✅ Advanced Training Complete. Best model saved!")

if __name__ == "__main__":
    start_training()