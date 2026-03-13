import tensorflow as tf
from src.config import load_params

def load_datasets():
    """Load train, val, test datasets"""
    params = load_params()
    
    # Training data
    data_train = tf.keras.utils.image_dataset_from_directory(
        params['data']['train_path'],
        shuffle=True,
        image_size=(params['data']['img_height'], params['data']['img_width']),
        batch_size=params['data']['batch_size'],
    )
    
    # Validation data
    data_val = tf.keras.utils.image_dataset_from_directory(
        params['data']['val_path'],
        shuffle=False,
        image_size=(params['data']['img_height'], params['data']['img_width']),
        batch_size=params['data']['batch_size'],
    )
    
    # Test data
    data_test = tf.keras.utils.image_dataset_from_directory(
        params['data']['test_path'],
        shuffle=False,
        image_size=(params['data']['img_height'], params['data']['img_width']),
        batch_size=params['data']['batch_size'],
    )
    
    class_names = data_train.class_names
    
    return data_train, data_val, data_test, class_names