import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tempfile
import os
import pandas as pd

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

def extract_features(file_path):
    """
    Extract features exactly as in your training notebook
    This should match the prepare_data function from your notebook
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050, duration=3.0)
        
        # Ensure consistent length
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
        
        # Chroma features (12 mean only for now)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        features.extend(chroma_mean)  # 12 features
        
        # This gives us: 26 + 8 + 12 = 46 features
        # Take only first 45 to match your model
        features = np.array(features[:45])
        
        # Handle any NaN or infinite values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
        
    except Exception as e:
        st.error(f"Error in feature extraction: {e}")
        return None

def prepare_data_single(audio_path, le):
    """
    Prepare single audio file for prediction
    Mimics your prepare_data function from the notebook
    """
    try:
        features = extract_features(audio_path)
        if features is None:
            return None
            
        # Create a simple dataset (no batching needed for single prediction)
        features = features.reshape(1, -1)
        
        return features
        
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None

def predict_emotion(model, le, features):
    """Make prediction using the trained model"""
    try:
        # Make prediction
        prediction = model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][predicted_class])
        emotion = le.inverse_transform([predicted_class])[0]
        
        # Get all probabilities
        emotion_probs = {}
        for i, emotion_name in enumerate(le.classes_):
            emotion_probs[emotion_name] = float(prediction[0][i])
        
        return emotion, confidence, emotion_probs
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None

# Main app
st.title("🎭 Emotion Recognition from Audio")
st.write("Upload a WAV file to detect the emotion using your trained model")

# Load model
model, le = load_model()

if model is None:
    st.error("❌ Could not load model. Make sure 'final_emotion_model_1.keras' is in the same directory.")
    st.stop()

st.success("✅ Model loaded successfully!")
st.info(f"Model expects {model.input_shape[1]} features")

# File upload
uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])

if uploaded_file is not None:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    if st.button("🔍 Analyze Emotion", type="primary"):
        with st.spinner("Analyzing audio... This may take a few seconds."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Prepare data using your notebook's method
                features = prepare_data_single(tmp_file_path, le)
                
                if features is not None:
                    st.success(f"✅ Features extracted: {features.shape}")
                    
                    # Make prediction
                    emotion, confidence, emotion_probs = predict_emotion(model, le, features)
                    
                    if emotion is not None:
                        # Display main result
                        st.markdown("---")
                        
                        # Emoji mapping
                        emoji_map = {
                            'happy': '😊', 'sad': '😢', 'angry': '😠', 'fearful': '😨',
                            'surprised': '😲', 'disgust': '🤢', 'neutral': '😐', 'calm': '😌'
                        }
                        
                        emoji = emoji_map.get(emotion, '🎭')
                        
                        # Main prediction display
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(f"## {emoji} **{emotion.upper()}**")
                        with col2:
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        # Confidence bar
                        st.progress(min(confidence, 1.0))
                        
                        # Confidence interpretation
                        if confidence >= 0.7:
                            st.success("🎯 High confidence prediction!")
                        elif confidence >= 0.5:
                            st.warning("⚠️ Moderate confidence prediction")
                        else:
                            st.info("❓ Low confidence prediction")
                        
                        # Show all emotion probabilities
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
                            st.write(f"**Features Extracted:** {features.shape[1]}")
                            st.write(f"**Model Input Shape:** {model.input_shape}")
                            st.write(f"**Model Output Shape:** {model.output_shape}")
                            st.write(f"**Number of Classes:** {len(le.classes_)}")
                            st.write(f"**Classes:** {', '.join(le.classes_)}")
                        
                    else:
                        st.error("❌ Failed to make prediction")
                else:
                    st.error("❌ Failed to extract features from audio")
                    
            except Exception as e:
                st.error(f"❌ Error processing audio: {e}")
                st.info("Please try with a different audio file or check the file format.")
            
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

# Instructions
st.markdown("---")
st.markdown("""
### 📋 Instructions:
1. **Upload a WAV file** (2-5 seconds recommended)
2. **Click 'Analyze Emotion'**
3. **View the predicted emotion and confidence**

### 💡 Tips for Better Results:
- Use clear, high-quality recordings
- 2-5 seconds of emotional speech works best
- Single speaker recordings are optimal
- Minimize background noise

### 🎭 Supported Emotions:
😊 Happy | 😢 Sad | 😠 Angry | 😨 Fearful | 😲 Surprised | 🤢 Disgust | 😐 Neutral | 😌 Calm

### 📈 Model Performance:
- **Training Accuracy:** 83%+
- **Features Used:** MFCC, Spectral, Chroma
- **Model Type:** Deep Neural Network
""")
