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

def extract_features(audio_file):
    """Simple feature extraction"""
    try:
        y, sr = librosa.load(audio_file, sr=22050, duration=3.0)
        
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
    except:
        return None, None

# Main app
st.title("🎭 Emotion Recognition")

# Load model
model, le = load_model()
if model is None:
    st.stop()

# File upload
uploaded_file = st.file_uploader("Upload WAV file", type=['wav'])

if uploaded_file:
    if st.button("Predict"):
        # Save file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Process
        features = extract_features(tmp_path)
        os.unlink(tmp_path)
        
        if features is not None:
            emotion, confidence = predict_emotion(model, le, features)
            if emotion:
                st.success(f"Emotion: {emotion}")
                st.info(f"Confidence: {confidence:.2f}")
            else:
                st.error("Prediction failed")
        else:
            st.error("Feature extraction failed")
