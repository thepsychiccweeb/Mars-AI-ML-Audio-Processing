import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import tempfile
import os
import pickle

# Configure page
st.set_page_config(page_title="Emotion Recognition", page_icon="🎭")

@st.cache_resource
def load_model_and_preprocessing():
    """Load model and preprocessing objects"""
    try:
        # Load model
        model = tf.keras.models.load_model('final_emotion_model_1.keras')
        
        # Load preprocessing objects
        with open('preprocessing_objects.pkl', 'rb') as f:
            preprocessing = pickle.load(f)
        
        return model, preprocessing
    except Exception as e:
        st.error(f"Error loading model/preprocessing: {e}")
        return None, None

def extract_simple_features(file_path, duration=3.0, sr=22050):
    """Extract exactly 45 features - EXACT copy from your notebook"""
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
        st.error(f"Error extracting features: {e}")
        return np.zeros(45)

def preprocess_features(features, preprocessing):
    """Apply the same preprocessing as training"""
    try:
        # Reshape for sklearn
        features = features.reshape(1, -1)
        
        # Apply feature selection (SelectKBest)
        features_selected = preprocessing['selector'].transform(features)
        
        # Apply scaling (StandardScaler)
        features_scaled = preprocessing['scaler'].transform(features_selected)
        
        return features_scaled
    except Exception as e:
        st.error(f"Error preprocessing features: {e}")
        return None

def predict_emotion(model, preprocessing, features):
    """Make prediction using exact same pipeline as training"""
    try:
        # Preprocess features
        processed_features = preprocess_features(features, preprocessing)
        
        if processed_features is None:
            return None, None, None
        
        # Make prediction
        prediction = model.predict(processed_features, verbose=0)
        
        # Get results
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        emotion = preprocessing['label_encoder'].classes_[predicted_class]
        
        # Get all probabilities
        emotion_probs = {}
        for i, emotion_name in enumerate(preprocessing['label_encoder'].classes_):
            emotion_probs[emotion_name] = float(prediction[0][i])
        
        return emotion, confidence, emotion_probs
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None

# Main app
st.title("🎭 Emotion Recognition from Audio")
st.write("Upload a WAV file to detect emotion using your trained model")

# Load model and preprocessing
model, preprocessing = load_model_and_preprocessing()

if model is None or preprocessing is None:
    st.error("❌ Could not load model or preprocessing objects.")
    st.info("Make sure these files are in the same directory:")
    st.write("- final_emotion_model_1.keras")
    st.write("- preprocessing_objects.pkl")
    st.stop()

st.success("✅ Model and preprocessing loaded successfully!")
st.info(f"Model expects {model.input_shape[1]} features after preprocessing")

# File upload
uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])

if uploaded_file is not None:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    if st.button("🎯 Predict Emotion", type="primary"):
        with st.spinner("Processing audio..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Extract features using your exact function
                features = extract_simple_features(tmp_file_path)
                
                st.success(f"✅ Extracted {len(features)} features")
                
                # Show feature stats
                with st.expander("🔍 Feature Statistics"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Min", f"{np.min(features):.4f}")
                    with col2:
                        st.metric("Max", f"{np.max(features):.4f}")
                    with col3:
                        st.metric("Mean", f"{np.mean(features):.4f}")
                    with col4:
                        st.metric("Std", f"{np.std(features):.4f}")
                
                # Make prediction
                emotion, confidence, emotion_probs = predict_emotion(model, preprocessing, features)
                
                if emotion is not None:
                    # Display results
                    st.markdown("---")
                    
                    # Emoji mapping
                    emoji_map = {
                        'happy': '😊', 'sad': '😢', 'angry': '😠', 'fearful': '😨',
                        'surprised': '😲', 'disgust': '🤢', 'neutral': '😐', 'calm': '😌'
                    }
                    
                    emoji = emoji_map.get(emotion, '🎭')
                    
                    # Main result
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"## {emoji} **{emotion.upper()}**")
                    with col2:
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                    
                    # Confidence bar
                    st.progress(min(confidence, 1.0))
                    
                    # Confidence interpretation
                    if confidence >= 0.8:
                        st.success("🎯 Very high confidence!")
                    elif confidence >= 0.6:
                        st.success("✅ High confidence!")
                    elif confidence >= 0.4:
                        st.warning("⚠️ Moderate confidence")
                    else:
                        st.info("❓ Low confidence")
                    
                    # All emotion probabilities
                    with st.expander("📊 All Emotion Probabilities"):
                        sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
                        
                        for i, (emo, prob) in enumerate(sorted_emotions):
                            emo_emoji = emoji_map.get(emo, '🎭')
                            if i == 0:  # Top prediction
                                st.markdown(f"**{emo_emoji} {emo.capitalize()}: {prob*100:.2f}%** 🏆")
                            else:
                                st.write(f"{emo_emoji} {emo.capitalize()}: {prob*100:.2f}%")
                    
                    # Technical details
                    with st.expander("🔧 Technical Details"):
                        st.write(f"**Raw Features:** {len(features)}")
                        st.write(f"**Selected Features:** {len(preprocessing['selected_indices'])}")
                        st.write(f"**Model Input Shape:** {model.input_shape}")
                        st.write(f"**Preprocessing Pipeline:** Feature Selection → Standard Scaling")
                        st.write(f"**Supported Emotions:** {', '.join(preprocessing['label_encoder'].classes_)}")
                
                else:
                    st.error("❌ Failed to make prediction")
                    
            except Exception as e:
                st.error(f"❌ Error processing audio: {e}")
            
            finally:
                # Clean up
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

# Instructions
st.markdown("---")
st.markdown("""
### 📋 How to Use:
1. **Upload a WAV file** (2-5 seconds recommended)
2. **Click 'Predict Emotion'**
3. **View the predicted emotion and confidence**

### 🎭 Supported Emotions:
😊 Happy | 😢 Sad | 😠 Angry | 😨 Fearful | 😲 Surprised | 🤢 Disgust | 😐 Neutral | 😌 Calm

### 📈 Model Info:
- **Training Accuracy:** 83%+
- **Features:** 45 audio features (MFCC, Spectral, Chroma, etc.)
- **Preprocessing:** Feature Selection + Standard Scaling
- **Architecture:** Deep Neural Network
""")
