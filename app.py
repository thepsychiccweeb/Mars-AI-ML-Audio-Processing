import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tempfile
import os

# Configure page
st.set_page_config(page_title="Emotion Recognition", page_icon="🎭")

@st.cache_resource
def load_model():
    """Load the saved model and create label encoder"""
    try:
        model = tf.keras.models.load_model('final_emotion_model_1.keras')
        le = LabelEncoder()
        all_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        le.fit(all_emotions)
        return model, le
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def is_valid_wav_file(file_bytes):
    """Check if file is a valid WAV file"""
    try:
        # Check WAV header
        if len(file_bytes) < 44:
            return False
        
        # Check RIFF header
        if file_bytes[:4] != b'RIFF':
            return False
            
        # Check WAVE format
        if file_bytes[8:12] != b'WAVE':
            return False
            
        return True
    except:
        return False

def extract_features(audio_file):
    """Simple feature extraction"""
    try:
        y, sr = librosa.load(audio_file, sr=22050, duration=3.0)
        
        if len(y) == 0:
            return None
        
        # Basic features only
        features = []
        
        # MFCC (26 features)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # Spectral features (8 features)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([np.mean(zcr), np.std(zcr)])
        
        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.std(rms)])
        
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.extend([np.mean(centroid), np.std(centroid)])
        
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.extend([np.mean(rolloff), np.std(rolloff)])
        
        # Chroma (11 features to make total 45)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1)[:11])
        
        return np.array(features[:45])
    
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return None

def predict_emotion(model, le, features):
    """Make prediction"""
    try:
        features = features.reshape(1, -1)
        prediction = model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[0][predicted_class])
        emotion = le.inverse_transform([predicted_class])[0]
        return emotion, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
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

# File upload with validation
uploaded_file = st.file_uploader(
    "Choose a WAV file", 
    type=['wav'],
    help="Only WAV audio files are supported"
)

if uploaded_file is not None:
    # Validate file
    file_bytes = uploaded_file.getvalue()
    
    if not is_valid_wav_file(file_bytes):
        st.error("❌ Invalid WAV file. Please upload a proper WAV audio file.")
        st.info("Make sure your file:")
        st.write("- Has .wav extension")
        st.write("- Is a proper WAV format (not renamed MP3 or other format)")
        st.write("- Is not corrupted")
    else:
        st.success(f"✅ Valid WAV file: {uploaded_file.name}")
        st.info(f"File size: {len(file_bytes)} bytes")
        
        # Don't use st.audio() as it's causing the error
        st.write("🎵 Audio file loaded (player disabled to prevent errors)")
        
        if st.button("🔍 Predict Emotion", type="primary"):
            with st.spinner("Analyzing audio..."):
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                
                try:
                    # Extract features
                    features = extract_features(tmp_path)
                    
                    if features is not None:
                        st.success(f"✅ Extracted {len(features)} features")
                        
                        # Make prediction
                        emotion, confidence = predict_emotion(model, le, features)
                        
                        if emotion is not None:
                            # Display result
                            st.markdown("---")
                            
                            # Emoji mapping
                            emoji_map = {
                                'happy': '😊', 'sad': '😢', 'angry': '😠', 'fearful': '😨',
                                'surprised': '😲', 'disgust': '🤢', 'neutral': '😐', 'calm': '😌'
                            }
                            
                            emoji = emoji_map.get(emotion, '🎭')
                            
                            # Results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"## {emoji} {emotion.capitalize()}")
                            with col2:
                                st.metric("Confidence", f"{confidence*100:.1f}%")
                            
                            # Confidence bar
                            st.progress(min(confidence, 1.0))
                            
                            # Confidence interpretation
                            if confidence > 0.7:
                                st.success("🎯 High confidence prediction!")
                            elif confidence > 0.5:
                                st.warning("⚠️ Medium confidence prediction")
                            else:
                                st.info("❓ Low confidence prediction")
                                
                        else:
                            st.error("❌ Failed to make prediction")
                    else:
                        st.error("❌ Failed to extract features from audio")
                        
                except Exception as e:
                    st.error(f"❌ Processing error: {e}")
                
                finally:
                    # Clean up
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

# Instructions
st.markdown("---")
st.markdown("""
### 📋 Instructions:
1. **Upload a WAV file** (make sure it's a real WAV file, not renamed MP3)
2. **Click 'Predict Emotion'**
3. **See the result**

### 💡 Tips:
- Use clear, 2-5 second audio clips
- Single speaker works best
- Avoid background noise
- Make sure the file is actually in WAV format

### 🎭 Supported Emotions:
😊 Happy | 😢 Sad | 😠 Angry | 😨 Fearful | 😲 Surprised | 🤢 Disgust | 😐 Neutral | 😌 Calm
""")
