import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import tempfile
import os
import pickle

# Configure page
st.set_page_config(
    page_title="Emotion Recognition", 
    page_icon="🎭",
    layout="wide"
)

def extract_simple_features(file_path, duration=3.0, sr=22050):
    """Extract exactly 45 features - matching training pipeline"""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        
        # Pad if shorter than expected duration
        if len(y) < duration * sr:
            y = np.pad(y, (0, int(duration * sr) - len(y)), 'constant')
        
        # 1. MFCC (20 features: 10 mean + 10 std)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # 2. Zero Crossing Rate (2 features)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # 3. Root Mean Square Energy (2 features)
        rmse = librosa.feature.rms(y=y)
        rmse_mean = np.mean(rmse)
        rmse_std = np.std(rmse)
        
        # 4. Spectral Centroid (2 features)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        sc_mean = np.mean(spectral_centroid)
        sc_std = np.std(spectral_centroid)
        
        # 5. Spectral Rolloff (2 features)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        sr_mean = np.mean(spectral_rolloff)
        sr_std = np.std(spectral_rolloff)
        
        # 6. Chroma Features (12 features - mean only)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # 7. Spectral Contrast (5 features)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(spectral_contrast, axis=1)[:5]
        
        # Combine all features (total: 45)
        features = np.concatenate([
            mfcc_mean,      # 10 features
            mfcc_std,       # 10 features
            [zcr_mean, zcr_std, rmse_mean, rmse_std, 
             sc_mean, sc_std, sr_mean, sr_std],  # 8 features
            chroma_mean,    # 12 features
            contrast_mean   # 5 features
        ])
        
        # Handle any NaN or infinite values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
        
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return np.zeros(45)

@st.cache_resource
def load_model_components():
    """Load model and preprocessing components"""
    try:
        # Try to load the complete package first
        try:
            with open('complete_model_package.pkl', 'rb') as f:
                package = pickle.load(f)
            return (
                package['model'], 
                package['selector'], 
                package['scaler'], 
                package['label_encoder']
            )
        except FileNotFoundError:
            # Fallback: load individual files
            model = tf.keras.models.load_model('final_emotion_model_1.keras')
            
            with open('preprocessing_objects.pkl', 'rb') as f:
                preprocessing = pickle.load(f)
            
            return (
                model,
                preprocessing['selector'],
                preprocessing['scaler'], 
                preprocessing['label_encoder']
            )
            
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        return None, None, None, None

def preprocess_features(features, selector, scaler):
    """Apply the same preprocessing as training"""
    try:
        # Reshape for sklearn (expects 2D array)
        features_reshaped = features.reshape(1, -1)
        
        # Apply feature selection
        features_selected = selector.transform(features_reshaped)
        
        # Apply scaling
        features_scaled = scaler.transform(features_selected)
        
        return features_scaled
        
    except Exception as e:
        st.error(f"Error preprocessing features: {str(e)}")
        return None

def predict_emotion(model, selector, scaler, label_encoder, audio_path):
    """Complete prediction pipeline"""
    try:
        # Step 1: Extract raw features
        raw_features = extract_simple_features(audio_path)
        
        if raw_features is None:
            return None, None, None, None
        
        # Step 2: Preprocess features
        processed_features = preprocess_features(raw_features, selector, scaler)
        
        if processed_features is None:
            return None, None, None, None
        
        # Step 3: Make prediction
        prediction = model.predict(processed_features, verbose=0)
        
        # Step 4: Extract results
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        emotion = label_encoder.classes_[predicted_class]
        
        # Get all emotion probabilities
        emotion_probs = {}
        for i, emotion_name in enumerate(label_encoder.classes_):
            emotion_probs[emotion_name] = float(prediction[0][i])
        
        # Debug information
        debug_info = {
            'raw_features_shape': raw_features.shape,
            'raw_features_stats': {
                'min': float(np.min(raw_features)),
                'max': float(np.max(raw_features)),
                'mean': float(np.mean(raw_features)),
                'std': float(np.std(raw_features))
            },
            'processed_features_shape': processed_features.shape,
            'processed_features_stats': {
                'min': float(np.min(processed_features)),
                'max': float(np.max(processed_features)),
                'mean': float(np.mean(processed_features)),
                'std': float(np.std(processed_features))
            }
        }
        
        return emotion, confidence, emotion_probs, debug_info
        
    except Exception as e:
        st.error(f"Error in prediction pipeline: {str(e)}")
        return None, None, None, None

def main():
    # App header
    st.title("🎭 Audio Emotion Recognition")
    st.markdown("Upload a WAV file to detect the emotion using deep learning")
    
    # Load model components
    with st.spinner("Loading model..."):
        model, selector, scaler, label_encoder = load_model_components()
    
    if model is None:
        st.error("❌ Failed to load model components")
        st.info("Required files:")
        st.write("- `complete_model_package.pkl` OR")
        st.write("- `final_emotion_model_1.keras` + `preprocessing_objects.pkl`")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Model info
    with st.expander("ℹ️ Model Information"):
        st.write(f"**Input Shape:** {model.input_shape}")
        st.write(f"**Output Shape:** {model.output_shape}")
        st.write(f"**Supported Emotions:** {', '.join(label_encoder.classes_)}")
        st.write(f"**Total Parameters:** {model.count_params():,}")
    
    # File upload
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Choose a WAV audio file", 
        type=['wav'],
        help="Upload a 2-5 second audio clip for best results"
    )
    
    if uploaded_file is not None:
        # Show file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Prediction button
        if st.button("🎯 Analyze Emotion", type="primary", use_container_width=True):
            with st.spinner("Analyzing audio... This may take a few seconds."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Make prediction
                    emotion, confidence, emotion_probs, debug_info = predict_emotion(
                        model, selector, scaler, label_encoder, tmp_file_path
                    )
                    
                    if emotion is not None:
                        # Display results
                        st.markdown("---")
                        st.markdown("## 🎯 Prediction Results")
                        
                        # Emotion mapping
                        emoji_map = {
                            'happy': '😊', 'sad': '😢', 'angry': '😠', 'fearful': '😨',
                            'surprised': '😲', 'disgust': '🤢', 'neutral': '😐', 'calm': '😌'
                        }
                        
                        emotion_emoji = emoji_map.get(emotion, '🎭')
                        
                        # Main result display
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.markdown(f"### {emotion_emoji} **{emotion.upper()}**")
                        
                        with col2:
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        with col3:
                            # Confidence level
                            if confidence >= 0.8:
                                st.success("Very High")
                            elif confidence >= 0.6:
                                st.success("High")
                            elif confidence >= 0.4:
                                st.warning("Moderate")
                            else:
                                st.error("Low")
                        
                        # Confidence bar
                        st.progress(min(confidence, 1.0))
                        
                        # All emotion probabilities
                        st.markdown("### 📊 All Emotion Probabilities")
                        
                        # Sort emotions by probability
                        sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
                        
                        # Create columns for better layout
                        cols = st.columns(2)
                        
                        for i, (emo, prob) in enumerate(sorted_emotions):
                            emo_emoji = emoji_map.get(emo, '🎭')
                            col_idx = i % 2
                            
                            with cols[col_idx]:
                                if i == 0:  # Highest probability
                                    st.markdown(f"**{emo_emoji} {emo.capitalize()}: {prob*100:.2f}%** 🏆")
                                else:
                                    st.write(f"{emo_emoji} {emo.capitalize()}: {prob*100:.2f}%")
                        
                        # Technical details
                        with st.expander("🔧 Technical Details"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Raw Features:**")
                                st.write(f"Shape: {debug_info['raw_features_shape']}")
                                st.write(f"Min: {debug_info['raw_features_stats']['min']:.4f}")
                                st.write(f"Max: {debug_info['raw_features_stats']['max']:.4f}")
                                st.write(f"Mean: {debug_info['raw_features_stats']['mean']:.4f}")
                                st.write(f"Std: {debug_info['raw_features_stats']['std']:.4f}")
                            
                            with col2:
                                st.write("**Processed Features:**")
                                st.write(f"Shape: {debug_info['processed_features_shape']}")
                                st.write(f"Min: {debug_info['processed_features_stats']['min']:.4f}")
                                st.write(f"Max: {debug_info['processed_features_stats']['max']:.4f}")
                                st.write(f"Mean: {debug_info['processed_features_stats']['mean']:.4f}")
                                st.write(f"Std: {debug_info['processed_features_stats']['std']:.4f}")
                    
                    else:
                        st.error("❌ Failed to analyze the audio file")
                        st.info("Please try with a different audio file")
                
                except Exception as e:
                    st.error(f"❌ Error processing audio: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
    
    # Instructions and tips
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Upload** a WAV audio file (2-5 seconds recommended)
    2. **Listen** to the audio using the player above
    3. **Click** 'Analyze Emotion' to get predictions
    4. **View** results with confidence scores
    """)
    
    st.markdown("### 💡 Tips for Better Results")
    st.markdown("""
    - Use clear, high-quality recordings
    - Single speaker works best
    - 2-5 seconds of emotional speech is optimal
    - Minimize background noise
    - Ensure the emotion is clearly expressed
    """)
    
    st.markdown("### 🎭 Supported Emotions")
    emotion_cols = st.columns(4)
    emotions_display = [
        ("😊", "Happy"), ("😢", "Sad"), ("😠", "Angry"), ("😨", "Fearful"),
        ("😲", "Surprised"), ("🤢", "Disgust"), ("😐", "Neutral"), ("😌", "Calm")
    ]
    
    for i, (emoji, emotion) in enumerate(emotions_display):
        with emotion_cols[i % 4]:
            st.markdown(f"**{emoji} {emotion}**")

if __name__ == "__main__":
    main()
