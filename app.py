import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tempfile
import os
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Emotion Recognition AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_encoder():
    """Load the trained model and label encoder with multiple fallback methods"""
    model = None
    le = None
    model_info = None
    
    # Method 1: Try SavedModel format
    try:
        st.info("🔄 Attempting to load SavedModel format...")
        model = tf.keras.models.load_model('emotion_model_savedmodel')
        st.success("✅ SavedModel loaded successfully!")
    except Exception as e:
        st.warning(f"SavedModel loading failed: {str(e)[:100]}...")
        
        # Method 2: Try loading from architecture + weights
        try:
            st.info("🔄 Attempting to load from architecture + weights...")
            with open('model_architecture.json', 'r') as json_file:
                model_json = json_file.read()
            model = tf.keras.models.model_from_json(model_json)
            model.load_weights('model_weights.h5')
            st.success("✅ Model loaded from architecture + weights!")
        except Exception as e2:
            st.warning(f"Architecture + weights loading failed: {str(e2)[:100]}...")
            
            # Method 3: Try original .keras file with compatibility mode
            try:
                st.info("🔄 Attempting to load original .keras file...")
                # Set compatibility mode
                tf.config.experimental.enable_op_determinism()
                model = tf.keras.models.load_model('final_emotion_model_1.keras', compile=False)
                # Recompile the model
                model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                st.success("✅ Original .keras model loaded with compatibility mode!")
            except Exception as e3:
                st.error(f"All model loading methods failed. Last error: {str(e3)[:200]}...")
                return None, None, None
    
    # Load label encoder
    try:
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        st.success("✅ Label encoder loaded!")
    except:
        st.warning("Label encoder file not found, creating new one...")
        le = LabelEncoder()
        all_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        le.fit(all_emotions)
    
    # Load model info
    try:
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
    except:
        if model:
            model_info = {
                'input_shape': model.input_shape,
                'output_shape': model.output_shape,
                'num_classes': len(le.classes_),
                'emotions': list(le.classes_),
                'total_params': model.count_params()
            }
    
    return model, le, model_info

def extract_features(audio_file, sr=22050):
    """Extract audio features (same as training pipeline)"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=sr, duration=3.0)
        
        # Pad or trim to ensure consistent length
        if len(y) < sr * 3:
            y = np.pad(y, (0, sr * 3 - len(y)), mode='constant')
        else:
            y = y[:sr * 3]
        
        features = []
        
        # MFCC features (26 features: 13 mean + 13 std)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # Spectral features
        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([np.mean(zcr), np.std(zcr)])
        
        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.std(rms)])
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        
        # Chroma features (24 features: 12 mean + 12 std)
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
    """Predict emotion from features"""
    try:
        # Reshape features for model input
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]
        
        # Get emotion label
        emotion = le.inverse_transform([predicted_class])[0]
        
        # Get all probabilities for visualization
        all_probabilities = prediction[0]
        emotion_probs = {le.inverse_transform([i])[0]: prob for i, prob in enumerate(all_probabilities)}
        
        return emotion, confidence, emotion_probs
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None

def create_probability_chart(emotion_probs):
    """Create a bar chart of emotion probabilities"""
    emotions = list(emotion_probs.keys())
    probabilities = [prob * 100 for prob in emotion_probs.values()]
    
    # Create color map
    colors = ['#ff6b6b' if prob == max(probabilities) else '#4ecdc4' for prob in probabilities]
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=probabilities,
            marker_color=colors,
            text=[f'{prob:.1f}%' for prob in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Emotion Prediction Probabilities",
        xaxis_title="Emotions",
        yaxis_title="Probability (%)",
        showlegend=False,
        height=400,
        font=dict(size=12)
    )
    
    return fig

def get_emotion_emoji(emotion):
    """Get emoji for emotion"""
    emoji_map = {
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'fearful': '😨',
        'surprised': '😲',
        'disgust': '🤢',
        'neutral': '😐',
        'calm': '😌'
    }
    return emoji_map.get(emotion.lower(), '🎭')

def get_confidence_class(confidence):
    """Get CSS class based on confidence level"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"

def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Emotion Recognition AI</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model and encoder
    with st.spinner("Loading AI model..."):
        model, le, model_info = load_model_and_encoder()
    
    if model is None or le is None:
        st.markdown("""
        <div class="error-box">
            <h3>❌ Model Loading Failed</h3>
            <p>Please ensure you have the model files in the correct location:</p>
            <ul>
                <li><code>emotion_model_savedmodel/</code> (preferred)</li>
                <li><code>model_architecture.json</code> + <code>model_weights.h5</code></li>
                <li><code>final_emotion_model_1.keras</code> (fallback)</li>
            </ul>
            <p><strong>Solution:</strong> Run the model converter script first:</p>
            <code>python convert_model.py</code>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    st.success("✅ AI model loaded successfully!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("📋 About")
        st.write("""
        This AI system can recognize emotions from audio files.
        
        **Supported Emotions:**
        - 😊 Happy
        - 😢 Sad  
        - 😠 Angry
        - 😨 Fearful
        - 😲 Surprised
        - 🤢 Disgust
        - 😐 Neutral
        - 😌 Calm
        """)
        
        st.header("📁 File Requirements")
        st.write("""
        - **Format:** WAV files (MP3 also supported)
        - **Duration:** Any (will be processed to 3 seconds)
        - **Quality:** Clear audio works best
        - **Language:** Works with any language/speech
        """)
        
        st.header("🔧 Model Info")
        if model_info:
            st.write(f"**Parameters:** {model_info.get('total_params', 'Unknown'):,}")
            st.write(f"**Input Features:** {model_info.get('input_shape', [None, 'Unknown'])[1]}")
            st.write(f"**Classes:** {model_info.get('num_classes', len(le.classes_))}")
            st.write(f"**Model Type:** Neural Network")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🎤 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3'],
            help="Upload a WAV or MP3 audio file to analyze emotions"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Play audio
            st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[1]}')
