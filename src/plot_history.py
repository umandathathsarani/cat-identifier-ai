import os
import matplotlib.pyplot as plt
import json

def plot_training_metrics(history_json_path="models/training_history.json"):
    """
    Loads the saved training logs and creates high-quality plots 
    for Loss and Accuracy across all training epochs.
    """
    print("📊 Loading training history logs...")
    
    if not os.path.exists(history_json_path):
        print(f"❌ History file not found at {history_json_path}")
        print("Once your training script finishes, ensure it saves the history dictionary as a JSON file!")
        return

    with open(history_json_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history['loss']) + 1)

    # Create a figure with two subplots side-by-side
    plt.figure(figsize=(14, 5))

    # 1. Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], label='Training Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)

    # 2. Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy', marker='x')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    os.makedirs("plots", exist_ok=True)
    output_plot_path = "plots/training_performance.png"
    plt.savefig(output_plot_path, dpi=300)
    plt.show()
    
    print(f"✅ Performance charts successfully saved to {output_plot_path}")

if __name__ == "__main__":
    plot_training_metrics()