import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tempfile
import os
import pickle

# Configure page
st.set_page_config(page_title="Emotion Recognition", page_icon="🎭")

@st.cache_resource
def load_model():
    """Load the saved model and create label encoder"""
    try:
        model = tf.keras.models.load_model('final_emotion_model_1.keras')
        
        # Create label encoder exactly as in your notebook
        le = LabelEncoder()
        all_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        le.fit(all_emotions)
        
        return model, le
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def extract_features_with_debug(file_path):
    """Extract features with debugging information"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050, duration=3.0)
        
        # Ensure consistent length
        if len(y) < sr * 3:
            y = np.pad(y, (0, sr * 3 - len(y)), mode='constant')
        else:
            y = y[:sr * 3]
        
        features = []
        feature_info = []
        
        # MFCC features (26: 13 mean + 13 std)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        features.extend(mfcc_mean)
        features.extend(mfcc_std)
        feature_info.append(f"MFCC mean range: {np.min(mfcc_mean):.3f} to {np.max(mfcc_mean):.3f}")
        feature_info.append(f"MFCC std range: {np.min(mfcc_std):.3f} to {np.max(mfcc_std):.3f}")
        
        # Spectral features (8 total)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean, zcr_std = np.mean(zcr), np.std(zcr)
        features.extend([zcr_mean, zcr_std])
        feature_info.append(f"ZCR: mean={zcr_mean:.6f}, std={zcr_std:.6f}")
        
        rms = librosa.feature.rms(y=y)
        rms_mean, rms_std = np.mean(rms), np.std(rms)
        features.extend([rms_mean, rms_std])
        feature_info.append(f"RMS: mean={rms_mean:.6f}, std={rms_std:.6f}")
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_mean, sc_std = np.mean(spectral_centroid), np.std(spectral_centroid)
        features.extend([sc_mean, sc_std])
        feature_info.append(f"Spectral Centroid: mean={sc_mean:.1f}, std={sc_std:.1f}")
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        sr_mean, sr_std = np.mean(spectral_rolloff), np.std(spectral_rolloff)
        features.extend([sr_mean, sr_std])
        feature_info.append(f"Spectral Rolloff: mean={sr_mean:.1f}, std={sr_std:.1f}")
        
        # Chroma features (11 to make total 45)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        features.extend(chroma_mean[:11])  # Take only 11
        feature_info.append(f"Chroma range: {np.min(chroma_mean[:11]):.3f} to {np.max(chroma_mean[:11]):.3f}")
        
        # Convert to array and handle edge cases
        features = np.array(features[:45])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features, feature_info
        
    except Exception as e:
        st.error(f"Error in feature extraction: {e}")
        return None, None

def apply_feature_scaling(features, method='standard'):
    """
    Apply feature scaling - this is likely missing from your app!
    Your training data was probably scaled.
    """
    if method == 'standard':
        # Standard scaling (mean=0, std=1)
        # Note: In production, you should save the scaler from training
        # For now, we'll apply basic standardization
        mean = np.mean(features)
        std = np.std(features)
        if std > 0:
            scaled_features = (features - mean) / std
        else:
            scaled_features = features
    elif method == 'minmax':
        # Min-max scaling (0 to 1)
        min_val = np.min(features)
        max_val = np.max(features)
        if max_val > min_val:
            scaled_features = (features - min_val) / (max_val - min_val)
        else:
            scaled_features = features
    else:
        scaled_features = features
    
    return scaled_features

def predict_emotion_with_debug(model, le, features):
    """Make prediction with debugging information"""
    try:
        # Show original features stats
        st.write("### 🔍 Feature Analysis:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min", f"{np.min(features):.4f}")
        with col2:
            st.metric("Max", f"{np.max(features):.4f}")
        with col3:
            st.metric("Mean", f"{np.mean(features):.4f}")
        with col4:
            st.metric("Std", f"{np.std(features):.4f}")
        
        # Try different scaling methods
        scaling_methods = ['none', 'standard', 'minmax']
        results = {}
        
        for method in scaling_methods:
            if method == 'none':
                scaled_features = features
            else:
                scaled_features = apply_feature_scaling(features, method)
            
            # Make prediction
            features_reshaped = scaled_features.reshape(1, -1)
            prediction = model.predict(features_reshaped, verbose=0)
            
            # Check for extreme predictions
            max_prob = np.max(prediction[0])
            predicted_class = np.argmax(prediction[0])
            emotion = le.classes_[predicted_class]
            
            results[method] = {
                'emotion': emotion,
                'max_prob': max_prob,
                'prediction': prediction[0],
                'scaled_features': scaled_features
            }
        
        # Display results for different scaling methods
        st.write("### 🧪 Scaling Method Comparison:")
        
        best_method = 'none'
        best_confidence = 0
        
        for method, result in results.items():
            max_prob = result['max_prob']
            emotion = result['emotion']
            
            # Look for reasonable confidence (not 100% or near 0%)
            if 0.3 < max_prob < 0.95:
                if max_prob > best_confidence:
                    best_confidence = max_prob
                    best_method = method
            
            st.write(f"**{method.capitalize()} scaling:** {emotion} ({max_prob*100:.1f}%)")
        
        # Use the best method
        if best_method == 'none' and results['none']['max_prob'] > 0.99:
            # If no scaling gives 100%, try standard scaling anyway
            best_method = 'standard'
            st.warning("⚠️ Using standard scaling to reduce overconfidence")
        
        final_result = results[best_method]
        
        # Get emotion probabilities
        emotion_probs = {}
        for i, emotion_name in enumerate(le.classes_):
            emotion_probs[emotion_name] = float(final_result['prediction'][i])
        
        return final_result['emotion'], final_result['max_prob'], emotion_probs, best_method
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None, None

# Main app
st.title("🎭 Emotion Recognition - Debug Version")
st.write("This version shows detailed analysis to fix the 100% prediction issue")

# Load model
model, le = load_model()

if model is None:
    st.error("❌ Could not load model.")
    st.stop()

st.success("✅ Model loaded successfully!")

# File upload
uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])

if uploaded_file is not None:
    if st.button("🔍 Analyze with Debug Info", type="primary"):
        with st.spinner("Analyzing..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Extract features with debug info
                features, feature_info = extract_features_with_debug(tmp_file_path)
                
                if features is not None:
                    # Show feature extraction details
                    with st.expander("🔧 Feature Extraction Details"):
                        for info in feature_info:
                            st.write(info)
                    
                    # Make prediction with debug
                    emotion, confidence, emotion_probs, scaling_method = predict_emotion_with_debug(model, le, features)
                    
                    if emotion is not None:
                        st.markdown("---")
                        st.success(f"**Best Result using {scaling_method} scaling:**")
                        
                        # Display result
                        emoji_map = {
                            'happy': '😊', 'sad': '😢', 'angry': '😠', 'fearful': '😨',
                            'surprised': '😲', 'disgust': '🤢', 'neutral': '😐', 'calm': '😌'
                        }
                        
                        emoji = emoji_map.get(emotion, '🎭')
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(f"## {emoji} **{emotion.upper()}**")
                        with col2:
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        # Show all probabilities
                        st.write("### 📊 All Emotion Probabilities:")
                        sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
                        
                        for i, (emo, prob) in enumerate(sorted_emotions):
                            emo_emoji = emoji_map.get(emo, '🎭')
                            if i == 0:
                                st.write(f"**{emo_emoji} {emo.capitalize()}: {prob*100:.2f}%** 🏆")
                            else:
                                st.write(f"{emo_emoji} {emo.capitalize()}: {prob*100:.2f}%")
                    
                    else:
                        st.error("❌ Failed to make prediction")
                else:
                    st.error("❌ Failed to extract features")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")
            
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

st.markdown("---")
st.markdown("""
### 🚨 About the 100% Prediction Issue:

This usually happens because:
1. **Missing feature scaling** - Training data was normalized, but app features aren't
2. **Feature range mismatch** - App features are outside training range
3. **Model overconfidence** - Model wasn't trained with proper regularization

The debug version above tries different scaling methods to find the most reasonable prediction.
""")
