import os
import tensorflow as tf

# Constants matching MobileNetV3 specifications
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def create_data_pipeline(data_dir="data/processed"):
    print("⚙️ Initializing TensorFlow Data Pipeline...")

    # 1. Load Training Data (80% of the dataset)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # 2. Load Validation Data (20% of the dataset)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"\n✅ Successfully loaded {len(class_names)} classes.")

    # 3. Define Data Augmentation (Only applied to training data)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    def prepare(ds, shuffle=False, augment=False):
        # Resize and rescale pixel values from [0, 255] to [0, 1]
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        ds = ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(1000)

        # Apply augmentation only to the training set
        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

        # Prefetch to overlap data preprocessing and model execution
        return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 4. Optimize Datasets
    print("🚀 Optimizing datasets for performance (caching & prefetching)...")
    train_ds = prepare(train_ds, shuffle=True, augment=True)
    val_ds = prepare(val_ds)

    return train_ds, val_ds, class_names

if __name__ == "__main__":
    # Test the pipeline
    train_dataset, val_dataset, classes = create_data_pipeline()
    
    # Fetch one single batch to verify shapes
    for image_batch, labels_batch in train_dataset.take(1):
        print("\n--- Pipeline Verification ---")
        print(f"Batch Image Shape: {image_batch.shape}")
        print(f"Batch Label Shape: {labels_batch.shape}")
        print(f"Pixel values scaled to: Min {tf.reduce_min(image_batch):.2f}, Max {tf.reduce_max(image_batch):.2f}")
        print("---------------------------")