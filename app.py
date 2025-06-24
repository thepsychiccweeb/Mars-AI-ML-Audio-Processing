import streamlit as st
import librosa
import numpy as np
import pickle
import tempfile
import os

st.title("🔧 DEBUG: Notebook vs App Comparison")

# Load debug data
@st.cache_resource
def load_debug_data():
    with open('debug_data.pkl', 'rb') as f:
        return pickle.load(f)

debug_data = load_debug_data()

st.success("✅ Debug data loaded!")

# Show notebook results
st.markdown("## 📓 Notebook Results (Expected)")
st.write(f"**File:** {debug_data['test_file_path']}")
st.write(f"**True Emotion:** {debug_data['true_emotion']}")
st.write(f"**Predicted:** {debug_data['predicted_emotion']}")
st.write(f"**Confidence:** {debug_data['confidence']:.4f}")

st.markdown("### Raw Features (Notebook)")
st.write(f"Shape: {debug_data['raw_features'].shape}")
st.write(f"First 10: {debug_data['raw_features'][:10]}")

st.markdown("### Scaled Features (Notebook)")
st.write(f"Shape: {debug_data['scaled_features'].shape}")
st.write(f"First 10: {debug_data['scaled_features'][:10]}")

st.markdown("---")

# Test with uploaded file
uploaded_file = st.file_uploader("Upload the SAME WAV file", type=['wav'])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        st.markdown("## 📱 App Results (Actual)")
        
        # Extract features in app
        def extract_simple_features(file_path, duration=3.0, sr=22050):
            y, sr = librosa.load(file_path, sr=sr, duration=duration)
            if len(y) < duration * sr:
                y = np.pad(y, (0, int(duration * sr) - len(y)), 'constant')
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)
            zcr_std = np.std(zcr)
            
            rmse = librosa.feature.rms(y=y)
            rmse_mean = np.mean(rmse)
            rmse_std = np.std(rmse)
            
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            sc_mean = np.mean(spectral_centroid)
            sc_std = np.std(spectral_centroid)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            sr_mean = np.mean(spectral_rolloff)
            sr_std = np.std(spectral_rolloff)
            
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_mean = np.mean(spectral_contrast, axis=1)[:5]
            
            features = np.concatenate([
                mfcc_mean, mfcc_std,
                [zcr_mean, zcr_std, rmse_mean, rmse_std, sc_mean, sc_std, sr_mean, sr_std],
                chroma_mean,
                contrast_mean
            ])
            
            return features
        
        # Extract features
        app_features = extract_simple_features(tmp_path)
        
        st.markdown("### Raw Features (App)")
        st.write(f"Shape: {app_features.shape}")
        st.write(f"First 10: {app_features[:10]}")
        
        # Compare raw features
        feature_diff = np.abs(debug_data['raw_features'] - app_features)
        st.write(f"**Max difference in raw features:** {np.max(feature_diff):.6f}")
        
        if np.max(feature_diff) > 0.001:
            st.error("❌ RAW FEATURES DON'T MATCH!")
            st.write("Notebook features:", debug_data['raw_features'][:5])
            st.write("App features:", app_features[:5])
        else:
            st.success("✅ Raw features match!")
        
        # Apply preprocessing
        app_features_reshaped = app_features.reshape(1, -1)
        app_features_selected = debug_data['selector'].transform(app_features_reshaped)
        app_features_scaled = debug_data['scaler'].transform(app_features_selected)
        
        st.markdown("### Scaled Features (App)")
        st.write(f"Shape: {app_features_scaled.shape}")
        st.write(f"First 10: {app_features_scaled[0][:10]}")
        
        # Compare scaled features
        scaled_diff = np.abs(debug_data['scaled_features'] - app_features_scaled[0])
        st.write(f"**Max difference in scaled features:** {np.max(scaled_diff):.6f}")
        
        if np.max(scaled_diff) > 0.001:
            st.error("❌ SCALED FEATURES DON'T MATCH!")
        else:
            st.success("✅ Scaled features match!")
        
        # Make prediction
        app_prediction = debug_data['model'].predict(app_features_scaled, verbose=0)
        app_predicted_class = np.argmax(app_prediction[0])
        app_predicted_emotion = debug_data['label_encoder'].classes_[app_predicted_class]
        app_confidence = app_prediction[0][app_predicted_class]
        
        st.write(f"**App Predicted:** {app_predicted_emotion}")
        st.write(f"**App Confidence:** {app_confidence:.4f}")
        
        # Compare predictions
        pred_diff = np.abs(debug_data['prediction'] - app_prediction[0])
        st.write(f"**Max difference in predictions:** {np.max(pred_diff):.6f}")
        
        if debug_data['predicted_emotion'] == app_predicted_emotion:
            st.success("✅ PREDICTIONS MATCH!")
        else:
            st.error("❌ PREDICTIONS DON'T MATCH!")
            st.write("Notebook prediction:", debug_data['prediction'])
            st.write("App prediction:", app_prediction[0])
    
    finally:
        os.unlink(tmp_path)
