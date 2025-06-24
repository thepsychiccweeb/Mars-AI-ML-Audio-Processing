import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(page_title="Emotion Recognition", page_icon="🎭")

@st.cache_resource
def load_model():
    """Load the saved model and create label encoder"""
    try:
        model = tf.keras.models.load_model('final_emotion_model_1.keras')
        
        # Create label encoder (same as in your notebook)
        le = LabelEncoder()
        all_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        le.fit(all_emotions)
        
        return model, le
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def extract_features(audio_file):
    """Extract features - robust version"""
    try:
        # Load audio with error handling
        try:
            y, sr = librosa.load(audio_file, sr=22050, duration=3.0)
        except Exception as e:
            st.error(f"Error loading audio file: {e}")
            return None
        
        # Check if audio is valid
        if len(y) == 0:
            st.error("Audio file is empty or corrupted")
            return None
        
        # Pad or trim to ensure consistent length
        target_length = sr * 3  # 3 seconds
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        
        features = []
        
        try:
            # MFCC features (26: 13 mean + 13 std)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            if mfccs.size > 0:
                features.extend(np.mean(mfccs, axis=1))  # 13 features
                features.extend(np.std(mfccs, axis=1))   # 13 features
            else:
                features.extend([0] * 26)  # fallback
        except:
            features.extend([0] * 26)  # fallback
        
        try:
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features.extend([np.mean(zcr), np.std(zcr)])  # 2 features
        except:
            features.extend([0, 0])
        
        try:
            # RMS energy
            rms = librosa.feature.rms(y=y)
            features.extend([np.mean(rms), np.std(rms)])  # 2 features
        except:
            features.extend([0, 0])
        
        try:
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])  # 2 features
        except:
            features.extend([0, 0])
        
        try:
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])  # 2 features
        except:
            features.extend([0, 0])
        
        try:
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            if chroma.size > 0:
                chroma_mean = np.mean(chroma, axis=1)
                features.extend(chroma_mean[:11])  # Take only 11 to make total 45
            else:
                features.extend([0] * 11)
        except:
            features.extend([0] * 11)
        
        # Ensure exactly 45 features
        features = np.array(features[:45])
        if len(features) < 45:
            features = np.pad(features, (0, 45 - len(features)), mode='constant')
        
        # Replace any NaN or inf values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def predict_emotion(model, le, features):
    """Make prediction"""
    try:
        features = features.reshape(1, -1)
        prediction = model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][predicted_class])
        emotion = le.inverse_transform([predicted_class])[0]
        return emotion, confidence
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

# Main app
st.title("🎭 Emotion Recognition from Audio")
st.write("Upload a WAV file to detect the emotion")

# Load model
model, le = load_model()

if model is None:
    st.error("❌ Could not load model. Make sure 'final_emotion_model_1.keras' is in the same directory.")
    st.stop()

st.success("✅ Model loaded successfully!")

# File upload
uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])

if uploaded_file is not None:
    # Display file info
    st.write(f"**File:** {uploaded_file.name}")
    st.write(f"**Size:** {uploaded_file.size} bytes")
    
    # Play audio
    try:
        st.audio(uploaded_file)
    except Exception as e:
        st.warning(f"Could not play audio: {e}")
    
    if st.button("🔍 Predict Emotion"):
        with st.spinner("Analyzing..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Extract features
                features = extract_features(tmp_file_path)
                
                if features is not None:
                    st.success(f"✅ Extracted {len(features)} features")
                    
                    # Make prediction
                    emotion, confidence = predict_emotion(model, le, features)
                    
                    if emotion is not None:
                        # Display result
                        st.success(f"🎯 **Predicted Emotion: {emotion.upper()}**")
                        st.info(f"Confidence: {confidence*100:.1f}%")
                        
                        # Emoji mapping
                        emoji_map = {
                            'happy': '😊', 'sad': '😢', 'angry': '😠', 'fearful': '😨',
                            'surprised': '😲', 'disgust': '🤢', 'neutral': '😐', 'calm': '😌'
                        }
                        
                        emoji = emoji_map.get(emotion, '🎭')
                        st.markdown(f"## {emoji} {emotion.capitalize()}")
                        
                        # Confidence bar
                        st.progress(min(confidence, 1.0))  # Ensure value is <= 1.0
                        
                        # Show confidence level
                        if confidence > 0.7:
                            st.success("🎯 High confidence!")
                        elif confidence > 0.5:
                            st.warning("⚠️ Medium confidence")
                        else:
                            st.info("❓ Low confidence")
                        
                    else:
                        st.error("❌ Failed to make prediction")
                else:
                    st.error("❌ Failed to extract features from audio file")
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                st.info("Try uploading a different WAV file")
            
            finally:
                # Clean up
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

# Instructions
st.markdown("---")
st.markdown("""
**Instructions:**
1. Upload a WAV audio file (preferably 2-5 seconds long)
2. Click 'Predict Emotion' 
3. See the predicted emotion and confidence

**Tips:**
- Clear speech works better than noisy audio
- 2-5 second clips are optimal
- Single speaker recordings work best

**Supported Emotions:**
😊 Happy | 😢 Sad | 😠 Angry | 😨 Fearful | 😲 Surprised | 🤢 Disgust | 😐 Neutral | 😌 Calm
""")
