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
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_encoder():
    """Load the trained model and create label encoder"""
    try:
        # Load the saved model
        model = tf.keras.models.load_model('final_emotion_model_1.keras')
        
        # Recreate the label encoder (same as training)
        le = LabelEncoder()
        all_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        le.fit(all_emotions)
        
        return model, le
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure 'final_emotion_model_1.keras' is in the same directory as this app.")
        return None, None

def extract_features(audio_file, sr=22050):
    """Extract audio features (same as training pipeline)"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=sr, duration=3.0)  # Limit to 3 seconds
        
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
        model, le = load_model_and_encoder()
    
    if model is None or le is None:
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
        - **Format:** WAV files
        - **Duration:** Any (will be processed to 3 seconds)
        - **Quality:** Clear audio works best
        - **Language:** Works with any language/speech
        """)
        
        st.header("🔧 Model Info")
        if model:
            total_params = model.count_params()
            st.write(f"**Parameters:** {total_params:,}")
            st.write(f"**Input Features:** {model.input_shape[1]}")
            st.write(f"**Model Type:** Neural Network")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🎤 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose a WAV file",
            type=['wav'],
            help="Upload a WAV audio file to analyze emotions"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Play audio
            st.audio(uploaded_file, format='audio/wav')
            
            # Process button
            if st.button("🔍 Analyze Emotion", type="primary"):
                with st.spinner("Analyzing audio... This may take a few seconds."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Extract features
                        features = extract_features(tmp_file_path)
                        
                        if features is not None:
                            # Make prediction
                            emotion, confidence, emotion_probs = predict_emotion(model, le, features)
                            
                            if emotion is not None:
                                # Store results in session state
                                st.session_state.prediction_results = {
                                    'emotion': emotion,
                                    'confidence': confidence,
                                    'emotion_probs': emotion_probs
                                }
                        
                    except Exception as e:
                        st.error(f"Error processing audio: {e}")
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(tmp_file_path):
                            os.unlink(tmp_file_path)
    
    with col2:
        st.header("📊 Prediction Results")
        
        # Display results if available
        if hasattr(st.session_state, 'prediction_results'):
            results = st.session_state.prediction_results
            emotion = results['emotion']
            confidence = results['confidence']
            emotion_probs = results['emotion_probs']
            
            # Main prediction display
            emoji = get_emotion_emoji(emotion)
            confidence_class = get_confidence_class(confidence)
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="text-align: center; margin: 0;">
                    {emoji} <span style="text-transform: capitalize;">{emotion}</span>
                </h2>
                <p style="text-align: center; margin: 10px 0 0 0;">
                    Confidence: <span class="{confidence_class}">{confidence*100:.1f}%</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence interpretation
            if confidence >= 0.7:
                st.success("🎯 High confidence prediction!")
            elif confidence >= 0.5:
                st.warning("⚠️ Moderate confidence prediction.")
            else:
                st.error("❓ Low confidence prediction. Consider using clearer audio.")
            
            # Probability chart
            st.plotly_chart(create_probability_chart(emotion_probs), use_container_width=True)
            
            # Detailed probabilities
            with st.expander("📈 Detailed Probabilities"):
                sorted_probs = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
                for i, (emo, prob) in enumerate(sorted_probs):
                    emoji = get_emotion_emoji(emo)
                    if i == 0:  # Highest probability
                        st.markdown(f"**{emoji} {emo.capitalize()}: {prob*100:.2f}%** 🏆")
                    else:
                        st.write(f"{emoji} {emo.capitalize()}: {prob*100:.2f}%")
        
        else:
            st.info("👆 Upload a WAV file and click 'Analyze Emotion' to see results here.")
    
    # Additional features
    st.markdown("---")
    
    # Tips section
    with st.expander("💡 Tips for Better Results"):
        st.write("""
        **For optimal emotion recognition:**
        
        1. **Clear Audio**: Use high-quality recordings without background noise
        2. **Emotional Speech**: The audio should contain clear emotional expression
        3. **Appropriate Length**: 2-5 seconds of speech works best
        4. **Single Speaker**: Works best with one person speaking
        5. **Natural Expression**: Genuine emotions are recognized better than acted ones
        
        **Common Issues:**
        - Very quiet audio may not be processed correctly
        - Multiple speakers can confuse the model
        - Background music or noise can affect accuracy
        - Very short clips (< 1 second) may not have enough information
        """)
    
    # Sample audio section
    with st.expander("🎵 Try Sample Audio"):
        st.write("""
        Don't have an audio file? You can:
        1. Record yourself speaking with different emotions
        2. Use voice recording apps on your phone
        3. Convert other audio formats to WAV using online converters
        
        **Sample phrases to try:**
        - "I'm so happy today!" (Happy)
        - "This is really frustrating." (Angry)
        - "I'm feeling quite sad." (Sad)
        - "That really surprised me!" (Surprised)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🤖 Powered by Deep Learning | Built with Streamlit</p>
        <p>Model trained on emotional speech data using TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
