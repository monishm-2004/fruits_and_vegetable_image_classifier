import os
import json
import tensorflow as tf
from src.data_loader import load_datasets
from src.model import build_model
from src.config import load_params

def train():
    params = load_params()
    
    # Load data
    print("Loading datasets...")
    data_train, data_val, data_test, class_names = load_datasets()
    
    # Build model
    print("Building model...")
    model = build_model(num_classes=len(class_names))
    
    # Compile
    model.compile(
        optimizer=params['train']['optimizer'],
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Train
    print("Training model...")
    history = model.fit(
        data_train,
        validation_data=data_val,
        epochs=params['train']['epochs']
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/image_classifier.keras')
    print("Model saved to models/image_classifier.keras")
    
    # Save metrics
    os.makedirs('logs', exist_ok=True)
    metrics = {
        'train_loss': float(history.history['loss'][-1]),
        'train_accuracy': float(history.history['accuracy'][-1]),
        'val_loss': float(history.history['val_loss'][-1]),
        'val_accuracy': float(history.history['val_accuracy'][-1]),
    }
    with open('logs/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to logs/metrics.json")
    
    return history

if __name__ == '__main__':
    train()