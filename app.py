import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import tempfile
import os
import pickle

# Configure Streamlit page settings
st.set_page_config(
    page_title="Emotion Recognition App",
    page_icon="🎭",
    layout="centered"
)

@st.cache_resource
def load_exact_pipeline():
    """
    Load the exact trained model and preprocessing pipeline from notebook.
    
    Returns:
        tuple: (model, selector, scaler, label_encoder) or (None, None, None, None) if failed
    """
    try:
        with open('exact_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        
        return (
            pipeline['model'],
            pipeline['selector'], 
            pipeline['scaler'],
            pipeline['label_encoder']
        )
    except FileNotFoundError:
        st.error("❌ exact_pipeline.pkl not found!")
        st.info("Please run the extraction code in your notebook first.")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading pipeline: {e}")
        return None, None, None, None

def extract_simple_features(file_path, duration=3.0, sr=22050):
    """
    Extract 45 audio features from WAV file - EXACT copy from notebook.
    
    Features extracted:
    - MFCC: 20 features (10 mean + 10 std)
    - Zero Crossing Rate: 2 features (mean + std)
    - RMS Energy: 2 features (mean + std)
    - Spectral Centroid: 2 features (mean + std)
    - Spectral Rolloff: 2 features (mean + std)
    - Chroma: 12 features (mean only)
    - Spectral Contrast: 5 features (mean only)
    
    Args:
        file_path (str): Path to the audio file
        duration (float): Duration to load (seconds)
        sr (int): Sample rate
        
    Returns:
        numpy.ndarray: Array of 45 features
    """
    try:
        # Load audio file with specified sample rate and duration
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        
        # Pad audio with zeros if shorter than expected duration
        if len(y) < duration * sr:
            y = np.pad(y, (0, int(duration * sr) - len(y)), 'constant')
        
        # Extract MFCC features (Mel-frequency cepstral coefficients)
        # These capture the spectral characteristics of speech
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        mfcc_mean = np.mean(mfcc, axis=1)  # Mean across time frames
        mfcc_std = np.std(mfcc, axis=1)    # Standard deviation across time frames
        
        # Extract Zero Crossing Rate (measures how often signal crosses zero)
        # Higher ZCR typically indicates more noise or unvoiced speech
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # Extract RMS Energy (Root Mean Square - measures signal power)
        # Higher RMS indicates louder/more energetic audio
        rmse = librosa.feature.rms(y=y)
        rmse_mean = np.mean(rmse)
        rmse_std = np.std(rmse)
        
        # Extract Spectral Centroid (brightness of sound)
        # Higher values indicate brighter, more treble-heavy sounds
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_mean = np.mean(spectral_centroid)
        sc_std = np.std(spectral_centroid)
        
        # Extract Spectral Rolloff (frequency below which 85% of energy is contained)
        # Helps distinguish between harmonic and noise-like sounds
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        sr_mean = np.mean(spectral_rolloff)
        sr_std = np.std(spectral_rolloff)
        
        # Extract Chroma features (pitch class profiles)
        # Represents the 12 different pitch classes (C, C#, D, etc.)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)  # Only mean, no std
        
        # Extract Spectral Contrast (difference between peaks and valleys in spectrum)
        # Measures how much the spectrum varies across different frequency bands
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(spectral_contrast, axis=1)[:5]  # Take only first 5
        
        # Combine all features into single array (total: 45 features)
        features = np.concatenate([
            mfcc_mean,      # 10 features
            mfcc_std,       # 10 features
            [zcr_mean, zcr_std, rmse_mean, rmse_std, 
             sc_mean, sc_std, sr_mean, sr_std],  # 8 features
            chroma_mean,    # 12 features
            contrast_mean   # 5 features
        ])
        
        return features
        
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return np.zeros(45)  # Return zeros if extraction fails

def predict_emotion_exact(audio_path, model, selector, scaler, label_encoder):
    """
    Predict emotion using the EXACT same pipeline as the notebook.
    
    Pipeline steps:
    1. Extract 45 audio features
    2. Reshape features for sklearn compatibility
    3. Apply feature selection (SelectKBest)
    4. Apply standardization scaling
    5. Make model prediction
    6. Convert to emotion label
    
    Args:
        audio_path (str): Path to the audio file
        model: Trained TensorFlow model
        selector: Fitted SelectKBest feature selector
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder for emotion classes
        
    Returns:
        tuple: (predicted_emotion, confidence, all_emotion_probabilities) or (None, None, None)
    """
    try:
        # Step 1: Extract audio features (same as prepare_data function in notebook)
        features = extract_simple_features(audio_path)
        
        # Step 2: Reshape features for sklearn (convert 1D array to 2D for single sample)
        features_reshaped = features.reshape(1, -1)
        
        # Step 3: Apply feature selection (removes less important features)
        # Uses the same selector that was fitted on training data
        features_selected = selector.transform(features_reshaped)
        
        # Step 4: Apply standardization scaling (zero mean, unit variance)
        # Uses the same scaler that was fitted on training data
        features_scaled = scaler.transform(features_selected)
        
        # Step 5: Make prediction using the trained neural network model
        prediction = model.predict(features_scaled, verbose=0)
        
        # Step 6: Extract results from model output
        predicted_class = np.argmax(prediction[0])  # Get class with highest probability
        confidence = float(prediction[0][predicted_class])  # Get confidence score
        emotion = label_encoder.classes_[predicted_class]  # Convert to emotion name
        
        # Create dictionary of all emotion probabilities for detailed view
        all_emotions = {}
        for i, emo_name in enumerate(label_encoder.classes_):
            all_emotions[emo_name] = float(prediction[0][i])
        
        return emotion, confidence, all_emotions
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None, None

def main():
    """Main Streamlit application function."""
    
    # Display app title and description
    st.title("🎭 Audio Emotion Recognition")
    st.markdown("Upload a WAV file to detect emotions using your exact trained pipeline")
    
    # Load the trained model and preprocessing pipeline
    with st.spinner("Loading exact pipeline..."):
        model, selector, scaler, label_encoder = load_exact_pipeline()
    
    # Stop execution if pipeline failed to load
    if model is None:
        st.stop()
    
    st.success("✅ Exact pipeline loaded successfully!")
    
    # Display expandable section with technical pipeline information
    with st.expander("📊 Pipeline Information"):
        st.write(f"**Model Input Shape:** {model.input_shape}")
        st.write(f"**Model Parameters:** {model.count_params():,}")
        st.write(f"**Feature Selector:** {type(selector).__name__}")
        st.write(f"**Scaler:** {type(scaler).__name__}")
        st.write(f"**Supported Emotions:** {', '.join(label_encoder.classes_)}")
    
    st.markdown("---")
    
    # File upload widget for WAV files
    uploaded_file = st.file_uploader(
        "Choose a WAV audio file",
        type=['wav'],
        help="Upload the same file you tested in your notebook"
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Display audio player so user can listen to uploaded file
        st.audio(uploaded_file, format='audio/wav')
        
        # Main prediction button
        if st.button("🎯 Predict Emotion", type="primary"):
            with st.spinner("Using exact pipeline from notebook..."):
                # Save uploaded file to temporary location for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                try:
                    # Make prediction using the exact same pipeline as notebook
                    emotion, confidence, all_emotions = predict_emotion_exact(
                        temp_path, model, selector, scaler, label_encoder
                    )
                    
                    # Display results if prediction was successful
                    if emotion is not None:
                        # Create results section
                        st.markdown("---")
                        st.markdown("## 🎯 Prediction Results")
                        
                        # Emoji mapping for visual appeal
                        emotion_emojis = {
                            'happy': '😊', 'sad': '😢', 'angry': '😠', 
                            'fearful': '😨', 'surprised': '😲', 'disgust': '🤢',
                            'neutral': '😐', 'calm': '😌'
                        }
                        
                        emoji = emotion_emojis.get(emotion, '🎭')
                        
                        # Display main prediction result with confidence
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(f"### {emoji} **{emotion.upper()}**")
                        with col2:
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        # Visual confidence indicator (progress bar)
                        st.progress(min(confidence, 1.0))
                        
                        # Display all emotion probabilities in ranked order
                        st.markdown("### 📊 All Emotion Probabilities")
                        
                        # Sort emotions by probability (highest first)
                        sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
                        
                        # Display each emotion with its probability
                        for i, (emo, prob) in enumerate(sorted_emotions):
                            emo_emoji = emotion_emojis.get(emo, '🎭')
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                if i == 0:  # Highlight top prediction with trophy
                                    st.markdown(f"**{emo_emoji} {emo.capitalize()}** 🏆")
                                else:
                                    st.write(f"{emo_emoji} {emo.capitalize()}")
                            with col2:
                                st.write(f"{prob*100:.2f}%")
                        
                        # Provide confidence interpretation for user understanding
                        st.markdown("---")
                        if confidence >= 0.8:
                            st.success("🎯 **Very High Confidence** - Model is very sure!")
                        elif confidence >= 0.6:
                            st.success("✅ **High Confidence** - Good prediction!")
                        elif confidence >= 0.4:
                            st.warning("⚠️ **Moderate Confidence** - Somewhat uncertain")
                        else:
                            st.error("❓ **Low Confidence** - Model is unsure")
                    
                    else:
                        st.error("❌ Failed to analyze the audio file")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                
                finally:
                    # Clean up temporary file to prevent disk space issues
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
    
    # Display usage instructions
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. **Upload** the same WAV file you tested in your notebook
    2. **Click** "Predict Emotion" to use the exact same pipeline
    3. **Compare** results with your notebook predictions
    4. **Report** any differences for debugging
    """)
    
    # Display supported emotions with visual icons
    st.markdown("### 🎭 Supported Emotions")
    emotions_list = [
        ("😊", "Happy"), ("😢", "Sad"), ("😠", "Angry"), ("😨", "Fearful"),
        ("😲", "Surprised"), ("🤢", "Disgust"), ("😐", "Neutral"), ("😌", "Calm")
    ]
    
    # Display emotions in a 4-column grid layout
    cols = st.columns(4)
    for i, (emoji, emotion) in enumerate(emotions_list):
        with cols[i % 4]:
            st.markdown(f"**{emoji} {emotion}**")

# Run the main application
if __name__ == "__main__":
    main()
