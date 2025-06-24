
import librosa
import numpy as np

def extract_simple_features(file_path, duration=3.0, sr=22050):
    """EXACT copy from notebook"""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        
        # If audio is shorter than expected duration, pad with zeros
        if len(y) < duration * sr:
            y = np.pad(y, (0, int(duration * sr) - len(y)), 'constant')
        
        # 1. MFCC (20 features - mean + std of 10 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # 2. Zero Crossing Rate (2 features)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # 3. Root Mean Square Energy (2 features)
        rmse = librosa.feature.rms(y=y)
        rmse_mean = np.mean(rmse)
        rmse_std = np.std(rmse)
        
        # 4. Spectral Centroid (2 features)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_mean = np.mean(spectral_centroid)
        sc_std = np.std(spectral_centroid)
        
        # 5. Spectral Rolloff (2 features)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        sr_mean = np.mean(spectral_rolloff)
        sr_std = np.std(spectral_rolloff)
        
        # 6. Chroma Features (12 features - mean only)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # 7. Spectral Contrast (5 features)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(spectral_contrast, axis=1)[:5]
        
        # Combine all features to exactly 45
        features = np.concatenate([
            mfcc_mean, mfcc_std,           # 20 features (10+10)
            [zcr_mean, zcr_std, rmse_mean, rmse_std, sc_mean, sc_std, sr_mean, sr_std],  # 8 features
            chroma_mean,                   # 12 features
            contrast_mean                  # 5 features
        ])
        
        return features  # Total: 45 features (20+8+12+5)
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return np.zeros(45)

def preprocess_for_model(features, selector, scaler):
    """EXACT preprocessing from prepare_data function"""
    # Reshape for sklearn
    features_reshaped = features.reshape(1, -1)
    
    # Apply feature selection (SelectKBest)
    features_selected = selector.transform(features_reshaped)
    
    # Apply scaling (StandardScaler)
    features_scaled = scaler.transform(features_selected)
    
    return features_scaled

def predict_emotion_exact(file_path, model, selector, scaler, label_encoder):
    """EXACT prediction pipeline from notebook"""
    # Extract features
    features = extract_simple_features(file_path)
    
    # Preprocess
    processed_features = preprocess_for_model(features, selector, scaler)
    
    # Predict
    prediction = model.predict(processed_features, verbose=0)
    
    # Get results
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    emotion = label_encoder.classes_[predicted_class]
    
    return emotion, confidence, prediction[0]
