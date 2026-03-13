import tensorflow as tf
import numpy as np
import argparse
from src.config import load_params
from src.data_loader import load_datasets

def predict(image_path):
    params = load_params()
    
    # Load class names
    _, _, _, class_names = load_datasets()
    
    # Load model
    model = tf.keras.models.load_model('models/image_classifier.keras')
    
    # Load and preprocess image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(params['data']['img_height'], params['data']['img_width'])
    )
    img_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)
    
    # Predict
    predictions = model.predict(img_bat)
    score = tf.nn.softmax(predictions)
    
    predicted_class = class_names[np.argmax(score)]
    confidence = np.max(score) * 100
    
    print(f'Predicted: {predicted_class} (Confidence: {confidence:.2f}%)')
    return predicted_class, confidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()
    
    predict(args.image)