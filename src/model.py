from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from src.config import load_params

def build_model(num_classes):
    """Build CNN model"""
    params = load_params()
    
    model = Sequential([
        layers.Rescaling(1./255),
        layers.Conv2D(params['model']['conv_filters'][0], 
                      params['model']['conv_kernel'], 
                      padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(params['model']['conv_filters'][1], 
                      params['model']['conv_kernel'], 
                      padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(params['model']['conv_filters'][2], 
                      params['model']['conv_kernel'], 
                      padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(params['model']['dropout_rate']),
        layers.Dense(params['model']['dense_units'], activation='relu'),
        layers.Dense(num_classes)
    ])
    
    return model