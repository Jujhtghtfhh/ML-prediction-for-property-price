"""
Real Estate Price Prediction Web Application
Flask backend with ML prediction
"""

import os
import pickle
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['UPLOAD_FOLDER'] = '/home/claude/real_estate_app/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
MODEL_DIR = './models'

def load_models():
    """Load all trained models and metadata"""
    with open(os.path.join(MODEL_DIR, 'price_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(MODEL_DIR, 'city_encoder.pkl'), 'rb') as f:
        city_encoder = pickle.load(f)
    
    with open(os.path.join(MODEL_DIR, 'type_encoder.pkl'), 'rb') as f:
        type_encoder = pickle.load(f)
    
    with open(os.path.join(MODEL_DIR, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    return model, scaler, city_encoder, type_encoder, metadata

# Load models at startup
model, scaler, city_encoder, type_encoder, metadata = load_models()

def extract_simple_image_features(image_path):
    """Extract simple statistical features from an image"""
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((64, 64))
        img_array = np.array(img)
        
        features = []
        
        # Color statistics per channel
        for channel in range(3):
            channel_data = img_array[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
            ])
        
        # Overall brightness
        gray = np.mean(img_array, axis=2)
        features.extend([
            np.mean(gray),
            np.std(gray),
        ])
        
        # Color ratios
        r_mean = np.mean(img_array[:, :, 0])
        g_mean = np.mean(img_array[:, :, 1])
        b_mean = np.mean(img_array[:, :, 2])
        total = r_mean + g_mean + b_mean + 1e-6
        features.extend([
            r_mean / total,
            g_mean / total,
            b_mean / total,
        ])
        
        # Edge detection
        gray = gray.astype(np.float32)
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        features.extend([
            np.mean(gx),
            np.mean(gy),
            np.std(gx),
            np.std(gy),
        ])
        
        return np.array(features)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def process_uploaded_images(files):
    """Process uploaded images and extract features"""
    all_features = []
    
    for file in files[:5]:  # Limit to 5 images
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            features = extract_simple_image_features(filepath)
            if features is not None:
                all_features.append(features)
            
            # Clean up
            os.remove(filepath)
    
    if all_features:
        return np.mean(all_features, axis=0)
    return np.zeros(24)

def encode_location(location):
    """Encode location, handling unknown values"""
    try:
        return city_encoder.transform([location])[0]
    except:
        # Return mean encoding for unknown locations
        return 0

def encode_property_type(prop_type):
    """Encode property type, handling unknown values"""
    try:
        return type_encoder.transform([prop_type])[0]
    except:
        return 0

def prepare_features(data, image_features):
    """Prepare feature vector for prediction"""
    # Numeric features
    bedrooms = float(data.get('bedrooms', 0))
    bathrooms = float(data.get('bathrooms', 0))
    indoor_area = float(data.get('indoor_area', 0))
    outdoor_area = float(data.get('outdoor_area', 0))
    
    # Encode categorical
    city_encoded = encode_location(data.get('location', ''))
    type_encoded = encode_property_type(data.get('property_type', ''))
    
    # Create feature vector
    numeric_features = [bedrooms, bathrooms, indoor_area, outdoor_area, 
                        city_encoded, type_encoded]
    
    # Binary features
    selected_features = data.get('features', [])
    if isinstance(selected_features, str):
        selected_features = [selected_features]
    
    binary_features = []
    for feature in metadata['all_features']:
        binary_features.append(1 if feature in selected_features else 0)
    
    # Combine all features
    all_features = numeric_features + binary_features + list(image_features)
    
    return np.array(all_features).reshape(1, -1)

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html',
                         locations=metadata['cities'],
                         property_types=metadata['types'],
                         features=metadata['all_features'])

@app.route('/predict', methods=['POST'])
def predict():
    """Make price prediction"""
    try:
        # Get form data
        data = {
            'location': request.form.get('location', ''),
            'property_type': request.form.get('property_type', ''),
            'bedrooms': request.form.get('bedrooms', 0),
            'bathrooms': request.form.get('bathrooms', 0),
            'indoor_area': request.form.get('indoor_area', 0),
            'outdoor_area': request.form.get('outdoor_area', 0),
            'features': request.form.getlist('features')
        }
        
        # Process images
        images = request.files.getlist('images')
        image_features = process_uploaded_images(images)
        
        # Prepare features
        X = prepare_features(data, image_features)
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        
        # Ensure positive prediction
        prediction = max(prediction, 10000)
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'formatted': f"€{prediction:,.0f}"
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/metadata')
def get_metadata():
    """Get available options for the form"""
    return jsonify({
        'locations': metadata['cities'],
        'property_types': metadata['types'],
        'features': metadata['all_features']
    })

if __name__ == '__main__':
    print("Starting Real Estate Price Prediction Server...")
    print(f"Loaded model with {len(metadata['all_features'])} features")
    print(f"Available locations: {metadata['cities']}")
    print(f"Available property types: {metadata['types']}")
    app.run(host='0.0.0.0', port=5000, debug=True)
