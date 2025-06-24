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
        # Load model with custom objects if needed
        model = tf.keras.models.load_model('final_emotion_model_1.keras')
        
        # Create label encoder (same as in your notebook)
        le = LabelEncoder()
        all_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        le.fit(all_emotions)
        
        return model, le
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def extract_features(audio_file, sr=22050):
    """Extract features exactly like in your notebook"""
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=sr, duration=3.0)
        
        # Pad or trim to 3 seconds
        if len(y) < sr * 3:
            y = np.pad(y, (0, sr * 3 - len(y)), mode='constant')
        else:
            y = y[:sr * 3]
        
        features = []
        
        # MFCC features (26: 13 mean + 13 std)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # Spectral features (8 total)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([np.mean(zcr), np.std(zcr)])
        
        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.std(rms)])
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        
        # Chroma features (24: 12 mean + 12 std)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))
        
        # Spectral contrast (7 features)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend(np.mean(contrast, axis=1))
        
        # Tonnetz features (6 features)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features.extend(np.mean(tonnetz, axis=1))
        
        return np.array(features)
    
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def predict_emotion(model, le, features):
    """Make prediction"""
    try:
        features = features.reshape(1, -1)
        prediction = model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]
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
    st.audio(uploaded_file)
    
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
                        st.progress(confidence)
                        
                    else:
                        st.error("❌ Failed to make prediction")
                else:
                    st.error("❌ Failed to extract features")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")
            
            finally:
                # Clean up
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

# Instructions
st.markdown("---")
st.markdown("""
**Instructions:**
1. Upload a WAV audio file
2. Click 'Predict Emotion'
3. See the predicted emotion and confidence

**Supported Emotions:**
😊 Happy | 😢 Sad | 😠 Angry | 😨 Fearful | 😲 Surprised | 🤢 Disgust | 😐 Neutral | 😌 Calm
""")
