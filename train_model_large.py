"""
Real Estate Price Prediction Model Training
Optimized for large datasets (12000+ samples)
Uses tabular data and image features for prediction
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from PIL import Image
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

DATA_DIR = "./data" 


MODEL_DIR = "./models"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_IMAGES_PER_PROPERTY = 3
NUM_WORKERS = 4


CSV_PATH = os.path.join(DATA_DIR, "properties.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
DESCRIPTIONS_DIR = os.path.join(DATA_DIR, "descriptions")


def parse_price(price_str):
    if pd.isna(price_str) or price_str == 'NA' or price_str == '':
        return None
        
    price_str = str(price_str)
    price_str = re.sub(r'[^\d,.]', '', price_str)
    if ',' in price_str and '.' in price_str:
        if price_str.index(',') > price_str.index('.'):
            price_str = price_str.replace('.', '').replace(',', '.')
        else:
            price_str = price_str.replace(',', '')
    else:
        price_str = price_str.replace(',', '')
    try:
        return float(price_str)
    except:
        return None


def parse_numeric(val):
    if pd.isna(val) or val == 'NA' or val == '':
        return 0.0
    try:
        return float(val)
    except:
        return 0.0

#Extracts image data. RGB value statistics, brightness, colour ratios, edges.
def extract_simple_image_features(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((64, 64))
        img_array = np.array(img, dtype=np.float32)
        
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
        
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        features.extend([
            np.mean(gx),
            np.mean(gy),
            np.std(gx),
            np.std(gy),
        ])
        
        return np.array(features, dtype=np.float32)
    except Exception as e:
        return None


def get_property_image_features(reference):
    img_folder = os.path.join(IMAGES_DIR, reference)
    num_features = 24
    
    if not os.path.exists(img_folder):
        return np.zeros(num_features, dtype=np.float32)
    
    all_features = []
    image_files = sorted([f for f in os.listdir(img_folder) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    for img_file in image_files[:MAX_IMAGES_PER_PROPERTY]:
        img_path = os.path.join(img_folder, img_file)
        features = extract_simple_image_features(img_path)
        if features is not None:
            all_features.append(features)
    
    if all_features:
        return np.mean(all_features, axis=0)
    return np.zeros(num_features, dtype=np.float32)


def load_and_process_data():
    print("=" * 60)
    print("1. DUOMENŲ ĮKĖLIMAS")
    print("=" * 60)
    
    if not os.path.exists(CSV_PATH):
        print(f"KLAIDA: Nerastas failas {CSV_PATH}")
        print("Patikrinkite DATA_DIR konfigūraciją failo pradžioje.")
        return None, None, None, None
    
    print(f"Skaitomas: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, header=None, encoding='latin-1', on_bad_lines='skip')
    
    df.columns = ['reference', 'location', 'price', 'title', 'bedrooms', 
                  'bathrooms', 'indoor_area', 'outdoor_area', 'features']
    
    print(f"Įkelta eilučių: {len(df):,}")
    
    print("\nApdorojamos kainos...")
    df['price_numeric'] = df['price'].apply(parse_price)
    valid_prices = df['price_numeric'].notna().sum()
    print(f"Kainų skaičius: {valid_prices:,} ({valid_prices/len(df)*100:.1f}%)")
    
    df['bedrooms_num'] = df['bedrooms'].apply(parse_numeric)
    df['bathrooms_num'] = df['bathrooms'].apply(parse_numeric)
    df['indoor_area_num'] = df['indoor_area'].apply(parse_numeric)
    df['outdoor_area_num'] = df['outdoor_area'].apply(parse_numeric)
    
    df['city'] = df['location'].apply(
        lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown'
    )
    df['region'] = df['location'].apply(
        lambda x: str(x).split(',')[1].strip() if pd.notna(x) and ',' in str(x) else 'Unknown'
    )
    
    print("\nApdorojami požymiai...")
    all_features = set()
    for feat_str in df['features'].dropna():
        if pd.notna(feat_str) and feat_str != 'NA':
            features = [f.strip() for f in str(feat_str).split('|')]
            all_features.update(features)
    
    all_features = sorted([f for f in all_features if f and len(f) > 1])
    print(f"Unikalių požymių skaičius: {len(all_features)}")
    
    for feature in all_features:
        col_name = 'feat_' + re.sub(r'[^a-z0-9]', '_', feature.lower())
        df[col_name] = df['features'].apply(
            lambda x: 1 if pd.notna(x) and feature in str(x) else 0
        )
    
    property_types = df['title'].dropna().unique().tolist()
    locations = df['city'].unique().tolist()
    
    print(f"\nTurto tipai ({len(property_types)}): {property_types[:5]}...")
    print(f"Vietos ({len(locations)}): {locations[:5]}...")
    
    return df, all_features, property_types, locations


def extract_all_image_features(references):
    print("\n" + "=" * 60)
    print("3. NUOTRAUKŲ APDOROJIMAS")
    print("=" * 60)
    
    if not os.path.exists(IMAGES_DIR):
        print(f"Įspėjimas: Nerastas nuotraukų katalogas {IMAGES_DIR}")
        print("Bus naudojami nuliniai vaizdų požymiai.")
        return [np.zeros(24, dtype=np.float32) for _ in references]
        
    print(f"Naudojamos {NUM_WORKERS} gijos")
    print(f"Apdorojama {len(references):,} nekilnojamojo turto objektų:")
    
    image_features = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(get_property_image_features, ref): i 
                   for i, ref in enumerate(references)}
        
        results = [None] * len(references)
        completed = 0
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = np.zeros(24, dtype=np.float32)
            
            completed += 1
            if completed % 100 == 0 or completed == len(references):
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta = (len(references) - completed) / rate if rate > 0 else 0
                print(" " * 100, end = "\r")
                print(f"  Progresas: {completed:,}/{len(references):,} "
                      f"({completed/len(references)*100:.1f}%) "
                      f"ETA: {eta:.0f}s", end = "\r")
        
        image_features = results
    print()
    elapsed = time.time() - start_time
    print(f"Nuotraukų apdorojimas baigtas per {elapsed:.1f}s")
    
    return image_features


def prepare_training_data(df, all_features):
    """Prepare feature matrix for training"""
    print("\n" + "=" * 60)
    print("2. POŽYMIŲ PARUOŠIMAS")
    print("=" * 60)
    
    df_valid = df[df['price_numeric'].notna()].copy()
    print(f"Tinkamų įrašų su kainomis: {len(df_valid):,}")
    
    q_low = df_valid['price_numeric'].quantile(0.01)
    q_high = df_valid['price_numeric'].quantile(0.99)
    df_valid = df_valid[
        (df_valid['price_numeric'] >= q_low) & 
        (df_valid['price_numeric'] <= q_high)
    ]
    print(f"Po išskirčių pašalinimo: {len(df_valid):,}")
    
    city_encoder = LabelEncoder()
    type_encoder = LabelEncoder()
    
    df_valid['city_encoded'] = city_encoder.fit_transform(df_valid['city'])
    df_valid['type_encoded'] = type_encoder.fit_transform(df_valid['title'].fillna('Unknown'))
    
    print(f"Užkoduota vietų: {len(city_encoder.classes_)}")
    print(f"Užkoduota tipų: {len(type_encoder.classes_)}")
    
    numeric_cols = ['bedrooms_num', 'bathrooms_num', 'indoor_area_num', 
                    'outdoor_area_num', 'city_encoded', 'type_encoded']
    
    feature_cols = [col for col in df_valid.columns if col.startswith('feat_')]
    print(f"Binarinių požymių: {len(feature_cols)}")
    
    X_tabular = df_valid[numeric_cols + feature_cols].values
    
    references = df_valid['reference'].tolist()
    image_features = extract_all_image_features(references)
    X_images = np.array(image_features)
    
    X_combined = np.hstack([X_tabular, X_images])
    y = df_valid['price_numeric'].values
    
    print(f"\nGalutinė požymių matrica: {X_combined.shape}")
    print(f"Tikslų vektorius: {y.shape}")
    
    return X_combined, y, city_encoder, type_encoder, numeric_cols, feature_cols, df_valid


def train_model(X, y):
    print("\n" + "=" * 60)
    print("4. MODELIO APMOKYMAS")
    print("=" * 60)
    
    print("Normalizuojami požymiai...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Dalinami duomenys (testavimui: {TEST_SIZE*100:.0f}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Apmokymo imtis: {len(X_train):,}")
    print(f"  Testavimo imtis: {len(X_test):,}")
    
    print("\nApmokomas Gradient Boosting modelis...")
    start_time = time.time()
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=RANDOM_STATE,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"\nApmokymas baigtas per {train_time:.1f}s")
    
    print("\n" + "-" * 40)
    print("5. MODELIO ĮVERTINIMAS")
    print("-" * 40)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"\nApmokymo duomenys:")
    print(f"  R² = {train_r2:.4f}")
    print(f"  MAE = €{train_mae:,.0f}")
    print(f"  RMSE = €{train_rmse:,.0f}")
    
    print(f"\nTestavimo duomenys:")
    print(f"  R² = {test_r2:.4f}")
    print(f"  MAE = €{test_mae:,.0f}")
    print(f"  RMSE = €{test_rmse:,.0f}")
    
    print(f"\nPavyzdinės prognozės (pirmos 10):")
    for i in range(min(10, len(y_test))):
        error_pct = abs(y_test[i] - y_test_pred[i]) / y_test[i] * 100
        print(f"  Tikra: €{y_test[i]:>12,.0f} | "
              f"Prognozė: €{y_test_pred[i]:>12,.0f} | "
              f"Paklaida: {error_pct:>5.1f}%")
    
    print(f"\nSvarbiausi požymiai:")
    feature_importance = model.feature_importances_
    indices = np.argsort(feature_importance)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. Požymis #{idx}: {feature_importance[idx]:.4f}")
    
    return model, scaler, {
        'train_r2': train_r2, 'train_mae': train_mae, 'train_rmse': train_rmse,
        'test_r2': test_r2, 'test_mae': test_mae, 'test_rmse': test_rmse
    }


def save_models(model, scaler, city_encoder, type_encoder, all_features, 
                property_types, locations, feature_cols, metrics):
    print("\n" + "=" * 60)
    print("6. MODELIŲ IŠSAUGOJIMAS")
    print("=" * 60)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    with open(os.path.join(MODEL_DIR, 'price_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    print(f"Išsaugotas: price_model.pkl")
    
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Išsaugotas: scaler.pkl")
    
    with open(os.path.join(MODEL_DIR, 'city_encoder.pkl'), 'wb') as f:
        pickle.dump(city_encoder, f)
    print(f"Išsaugotas: city_encoder.pkl")
    
    with open(os.path.join(MODEL_DIR, 'type_encoder.pkl'), 'wb') as f:
        pickle.dump(type_encoder, f)
    print(f"Išsaugotas: type_encoder.pkl")
    
    metadata = {
        'all_features': all_features,
        'property_types': property_types,
        'locations': locations,
        'feature_cols': feature_cols,
        'cities': list(city_encoder.classes_),
        'types': list(type_encoder.classes_),
        'metrics': metrics
    }
    
    with open(os.path.join(MODEL_DIR, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Išsaugotas: metadata.pkl")
    
    print(f"\nVisi modeliai išsaugoti į: {MODEL_DIR}")


def main():
    print("\n" + "=" * 60)
    print("  NEKILNOJAMOJO TURTO KAINOS PROGNOZAVIMO")
    print("  MODELIO APMOKYMAS")
    print("=" * 60)
    print(f"\nDuomenų katalogas: {DATA_DIR}")
    print(f"Modelių katalogas: {MODEL_DIR}")
    
    total_start = time.time()
    
    result = load_and_process_data()
    if result[0] is None:
        return
    
    df, all_features, property_types, locations = result
    
    X, y, city_encoder, type_encoder, numeric_cols, feature_cols, df_valid = \
        prepare_training_data(df, all_features)
    
    model, scaler, metrics = train_model(X, y)
    
    save_models(model, scaler, city_encoder, type_encoder, all_features, 
                property_types, locations, feature_cols, metrics)
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("  APMOKYMAS BAIGTAS SĖKMINGAI!")
    print("=" * 60)
    print(f"\nBendras laikas: {total_time/60:.1f} min")
    print(f"Galutinis R² (test): {metrics['test_r2']:.4f}")
    print(f"Galutinis MAE (test): €{metrics['test_mae']:,.0f}")
    print("\nDabar galite paleisti app.py serverį!")


if __name__ == "__main__":
    main()
