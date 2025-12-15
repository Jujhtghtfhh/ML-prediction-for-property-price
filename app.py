import os
import pickle
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 
app.config['UPLOAD_FOLDER'] = '/home/claude/real_estate_app/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_DIR = './models'

def load_models():
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

model, scaler, city_encoder, type_encoder, metadata = load_models()

def extract_simple_image_features(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((64, 64))
        img_array = np.array(img)
        
        features = []
        
        for channel in range(3):
            channel_data = img_array[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75),
            ])
        
        gray = np.mean(img_array, axis=2)
        features.extend([
            np.mean(gray),
            np.std(gray),
        ])
        
        r_mean = np.mean(img_array[:, :, 0])
        g_mean = np.mean(img_array[:, :, 1])
        b_mean = np.mean(img_array[:, :, 2])
        total = r_mean + g_mean + b_mean + 1e-6
        features.extend([
            r_mean / total,
            g_mean / total,
            b_mean / total,
        ])
        
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
    all_features = []
    
    for file in files[:5]:
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            features = extract_simple_image_features(filepath)
            if features is not None:
                all_features.append(features)
            
            os.remove(filepath)
    
    if all_features:
        return np.mean(all_features, axis=0)
    return np.zeros(24)

def encode_location(location):
    try:
        return city_encoder.transform([location])[0]
    except:
        return 0

def encode_property_type(prop_type):
    try:
        return type_encoder.transform([prop_type])[0]
    except:
        return 0

def prepare_features(data, image_features):
    bedrooms = float(data.get('bedrooms', 0))
    bathrooms = float(data.get('bathrooms', 0))
    indoor_area = float(data.get('indoor_area', 0))
    outdoor_area = float(data.get('outdoor_area', 0))
    
    city_encoded = encode_location(data.get('location', ''))
    type_encoded = encode_property_type(data.get('property_type', ''))
    
    numeric_features = [bedrooms, bathrooms, indoor_area, outdoor_area, 
                        city_encoded, type_encoded]
    
    selected_features = data.get('features', [])
    if isinstance(selected_features, str):
        selected_features = [selected_features]
    
    binary_features = []
    for feature in metadata['all_features']:
        binary_features.append(1 if feature in selected_features else 0)
    
    all_features = numeric_features + binary_features + list(image_features)
    
    return np.array(all_features).reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html',
                         locations=metadata['cities'],
                         property_types=metadata['types'],
                         features=metadata['all_features'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'location': request.form.get('location', ''),
            'property_type': request.form.get('property_type', ''),
            'bedrooms': request.form.get('bedrooms', 0),
            'bathrooms': request.form.get('bathrooms', 0),
            'indoor_area': request.form.get('indoor_area', 0),
            'outdoor_area': request.form.get('outdoor_area', 0),
            'features': request.form.getlist('features')
        }
        
        images = request.files.getlist('images')
        image_features = process_uploaded_images(images)
        
        X = prepare_features(data, image_features)
        
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        
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
