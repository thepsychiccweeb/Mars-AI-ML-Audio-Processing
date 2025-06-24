import streamlit as st
import librosa
import numpy as np
import pickle
import tempfile
import os

st.title("🐛 Bug Hunter - Always Happy Problem")

# Load test data
with open('simple_test.pkl', 'rb') as f:
    test_data = pickle.load(f)

st.write("**Notebook Result:**", test_data['emotion'])

uploaded_file = st.file_uploader("Upload WAV", type=['wav'])

if uploaded_file:
    # Save file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    # EXACT same function as notebook
    def extract_simple_features(file_path, duration=3.0, sr=22050):
        try:
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
            
        except Exception as e:
            st.error(f"Feature extraction error: {e}")
            return None
    
    if st.button("Test"):
        # Extract features
        app_features = extract_simple_features(tmp_path)
        
        if app_features is None:
            st.error("Feature extraction failed!")
        else:
            st.write("**App Features (first 5):**", app_features[:5])
            st.write("**Notebook Features (first 5):**", test_data['raw_features'][:5])
            
            # Check if features match
            diff = np.abs(app_features - test_data['raw_features'])
            max_diff = np.max(diff)
            st.write(f"**Max difference:** {max_diff}")
            
            if max_diff > 0.01:
                st.error("❌ FEATURES DON'T MATCH!")
                st.write("This is why predictions are wrong!")
                
                # Show where they differ most
                worst_idx = np.argmax(diff)
                st.write(f"Worst difference at index {worst_idx}:")
                st.write(f"Notebook: {test_data['raw_features'][worst_idx]}")
                st.write(f"App: {app_features[worst_idx]}")
                
            else:
                st.success("✅ Features match!")
                
                # Test preprocessing
                app_reshaped = app_features.reshape(1, -1)
                app_selected = test_data['selector'].transform(app_reshaped)
                app_scaled = test_data['scaler'].transform(app_selected)
                
                st.write("**App Scaled (first 5):**", app_scaled[0][:5])
                st.write("**Notebook Scaled (first 5):**", test_data['scaled_features'][:5])
                
                # Test prediction
                app_pred = test_data['model'].predict(app_scaled, verbose=0)
                app_emotion = test_data['label_encoder'].classes_[np.argmax(app_pred[0])]
                
                st.write("**App Prediction:**", app_emotion)
                
                if app_emotion == test_data['emotion']:
                    st.success("✅ PREDICTIONS MATCH!")
                else:
                    st.error("❌ Still wrong prediction!")
                    st.write("App probs:", app_pred[0])
                    st.write("Notebook probs:", test_data['prediction'])
    
    # Cleanup
    os.unlink(tmp_path)
