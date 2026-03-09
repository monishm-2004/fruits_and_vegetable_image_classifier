import tensorflow as tf
import argparse
from data_loader import load_datasets
from model import build_model
from config import load_params

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
    model.save('models/image_classifier.keras')
    print("Model saved to models/image_classifier.keras")
    
    return history

if __name__ == '__main__':
    train()