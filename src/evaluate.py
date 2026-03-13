import os
import json
import argparse
import tensorflow as tf
from src.data_loader import load_datasets


def evaluate(model_path: str, split: str = "test"):
    # Load datasets
    _, data_val, data_test, _ = load_datasets()

    # Choose split
    dataset = data_val if split == "val" else data_test

    # Load trained model
    model = tf.keras.models.load_model(model_path)

    # Evaluate
    loss, accuracy = model.evaluate(dataset, verbose=1)
    print(f"{split.upper()} Loss: {loss:.4f}")
    print(f"{split.upper()} Accuracy: {accuracy:.4f}")

    # Save metrics
    os.makedirs('logs', exist_ok=True)
    with open('logs/test_metrics.json', 'w') as f:
        json.dump({'test_loss': float(loss), 'test_accuracy': float(accuracy)}, f, indent=4)
    print("Test metrics saved to logs/test_metrics.json")

    return loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained image classification model.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/image_classifier.keras",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate",
    )
    args = parser.parse_args()

    evaluate(model_path=args.model_path, split=args.split)