import os
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import io
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load the trained model
def load_model():
    model_path = 'models/digit_classifier_final.h5'
    backup_model_path = 'models/best_model.h5'
    
    if os.path.exists(model_path):
        logger.info("Loading model from digit_classifier_final.h5")
        return tf.keras.models.load_model(model_path)
    elif os.path.exists(backup_model_path):
        logger.info("Loading model from best_model.h5")
        return tf.keras.models.load_model(backup_model_path)
    else:
        logger.error("No model found! Please run train_model.py first.")
        return None

model = load_model()

def preprocess_image(image_data):
    """Preprocess image for model prediction"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Invert colors (the model was trained on white digits on black background)
        # If your drawing is black on white, uncomment the next line:
        img_array = 1.0 - img_array
        
        # Add batch and channel dimensions
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {'digit': int(idx), 'confidence': float(predictions[0][idx] * 100)}
            for idx in top_3_indices
        ]
        
        logger.info(f"Predicted digit: {predicted_class} with confidence: {confidence:.2f}%")
        
        return jsonify({
            'predicted_digit': int(predicted_class),
            'confidence': confidence,
            'top_3_predictions': top_3_predictions,
            'all_probabilities': predictions[0].tolist()
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("\n🚀 Starting Digit Classifier Web Application...")
    print("=" * 50)
    if model is None:
        print("⚠️  WARNING: No model found!")
        print("Please run 'python train_model.py' first to train the model.")
    else:
        print("✅ Model loaded successfully!")
    print("=" * 50)
    print("\n🌐 Open your browser and go to: http://localhost:5000")
    print("✏️  Draw a digit and click 'Predict Digit' to test the model")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)