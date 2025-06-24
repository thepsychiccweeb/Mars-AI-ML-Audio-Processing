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
        
        # Create label encoder (same as in your notebook)
        le = LabelEncoder()
        all_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        le.fit(all_emotions)
        
        return model, le
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def extract_features(audio_file):
    """Extract features - simplified to match model input (45 features)"""
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=22050, duration=3.0)
        
        features = []
        
        # MFCC features (26: 13 mean + 13 std)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))  # 13 features
        features.extend(np.std(mfccs, axis=1))   # 13 features
        
        # Basic spectral features (8 features)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([np.mean(zcr), np.std(zcr)])  # 2 features
        
        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.std(rms)])  # 2 features
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])  # 2 features
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])  # 2 features
        
        # Chroma features (12 mean only to reach 45 total)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))  # 12 features
        
        # Total: 26 + 8 + 12 = 46 features, remove 1 to get 45
        features = np.array(features[:45])
        
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
        confidence = float(prediction[0][predicted_class])  # Convert to Python float
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
                        
                        # Confidence bar (convert to Python float)
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
