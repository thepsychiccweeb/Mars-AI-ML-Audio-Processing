import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import tempfile
import os
import pickle

# Page configuration
st.set_page_config(
    page_title="Emotion Recognition App",
    page_icon="🎭",
    layout="centered"
)

@st.cache_resource
def load_model_pipeline():
    """Load the exact model and preprocessing pipeline"""
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
    """Extract features - EXACT copy from notebook"""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        
        # Pad if shorter
        if len(y) < duration * sr:
            y = np.pad(y, (0, int(duration * sr) - len(y)), 'constant')
        
        # 1. MFCC features (20 total: 10 mean + 10 std)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # 2. Zero Crossing Rate (2 features)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # 3. RMS Energy (2 features)
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
        
        # 6. Chroma features (12 features - mean only)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # 7. Spectral Contrast (5 features)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(spectral_contrast, axis=1)[:5]
        
        # Combine all features (total: 45)
        features = np.concatenate([
            mfcc_mean,      # 10
            mfcc_std,       # 10
            [zcr_mean, zcr_std, rmse_mean, rmse_std, 
             sc_mean, sc_std, sr_mean, sr_std],  # 8
            chroma_mean,    # 12
            contrast_mean   # 5
        ])
        
        return features
        
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return np.zeros(45)

def predict_emotion(audio_path, model, selector, scaler, label_encoder):
    """Exact prediction pipeline from notebook"""
    try:
        # Step 1: Extract features (same as prepare_data)
        features = extract_simple_features(audio_path)
        
        # Step 2: Reshape for sklearn
        features_reshaped = features.reshape(1, -1)
        
        # Step 3: Apply feature selection (same as prepare_data)
        features_selected = selector.transform(features_reshaped)
        
        # Step 4: Apply scaling (same as prepare_data)
        features_scaled = scaler.transform(features_selected)
        
        # Step 5: Model prediction
        prediction = model.predict(features_scaled, verbose=0)
        
        # Step 6: Extract results
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        emotion = label_encoder.classes_[predicted_class]
        
        # All emotion probabilities
        all_emotions = {}
        for i, emo_name in enumerate(label_encoder.classes_):
            all_emotions[emo_name] = float(prediction[0][i])
        
        return emotion, confidence, all_emotions
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None, None

def main():
    # App title
    st.title("🎭 Audio Emotion Recognition")
    st.markdown("Upload a WAV file to detect emotions using your trained model")
    
    # Load model pipeline
    with st.spinner("Loading model pipeline..."):
        model, selector, scaler, label_encoder = load_model_pipeline()
    
    if model is None:
        st.stop()
    
    st.success("✅ Model pipeline loaded successfully!")
    
    # Show model info
    with st.expander("📊 Model Information"):
        st.write(f"**Model Input Shape:** {model.input_shape}")
        st.write(f"**Number of Parameters:** {model.count_params():,}")
        st.write(f"**Supported Emotions:** {', '.join(label_encoder.classes_)}")
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a WAV audio file",
        type=['wav'],
        help="Upload a clear audio file (2-5 seconds recommended)"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Show audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Prediction button
        if st.button("🎯 Analyze Emotion", type="primary"):
            with st.spinner("Analyzing audio..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                try:
                    # Make prediction using exact pipeline
                    emotion, confidence, all_emotions = predict_emotion(
                        temp_path, model, selector, scaler, label_encoder
                    )
                    
                    if emotion is not None:
                        # Display results
                        st.markdown("---")
                        st.markdown("## 🎯 Results")
                        
                        # Emotion emojis
                        emotion_emojis = {
                            'happy': '😊', 'sad': '😢', 'angry': '😠', 
                            'fearful': '😨', 'surprised': '😲', 'disgust': '🤢',
                            'neutral': '😐', 'calm': '😌'
                        }
                        
                        emoji = emotion_emojis.get(emotion, '🎭')
                        
                        # Main result
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### {emoji} **{emotion.upper()}**")
                        with col2:
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        # Confidence indicator
                        confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                        st.markdown(f"<div style='background-color: {confidence_color}; height: 10px; width: {confidence*100}%; border-radius: 5px;'></div>", unsafe_allow_html=True)
                        
                        # All emotion probabilities
                        st.markdown("### 📊 All Emotion Probabilities")
                        
                        # Sort by probability
                        sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
                        
                        for i, (emo, prob) in enumerate(sorted_emotions):
                            emo_emoji = emotion_emojis.get(emo, '🎭')
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                if i == 0:  # Top prediction
                                    st.markdown(f"**{emo_emoji} {emo.capitalize()}** 🏆")
                                else:
                                    st.write(f"{emo_emoji} {emo.capitalize()}")
                            with col2:
                                st.write(f"{prob*100:.2f}%")
                        
                        # Confidence interpretation
                        st.markdown("---")
                        if confidence >= 0.8:
                            st.success("🎯 **Very High Confidence** - The model is very sure about this prediction!")
                        elif confidence >= 0.6:
                            st.success("✅ **High Confidence** - The model is confident about this prediction.")
                        elif confidence >= 0.4:
                            st.warning("⚠️ **Moderate Confidence** - The prediction might be uncertain.")
                        else:
                            st.error("❓ **Low Confidence** - The model is not sure. Try a clearer audio file.")
                    
                    else:
                        st.error("❌ Failed to analyze the audio file")
                
                except Exception as e:
                    st.error(f"❌ Error processing audio: {str(e)}")
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
    
    # Instructions
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Upload** a WAV audio file using the file uploader above
    2. **Listen** to your audio using the built-in player
    3. **Click** "Analyze Emotion" to get the prediction
    4. **View** the results with confidence scores
    """)
    
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - Use clear, high-quality audio recordings
    - Keep recordings between 2-5 seconds
    - Ensure the speaker clearly expresses the emotion
    - Minimize background noise
    - Single speaker works better than multiple speakers
    """)
    
    st.markdown("### 🎭 Supported Emotions")
    emotions_display = [
        ("😊", "Happy"), ("😢", "Sad"), ("😠", "Angry"), ("😨", "Fearful"),
        ("😲", "Surprised"), ("🤢", "Disgust"), ("😐", "Neutral"), ("😌", "Calm")
    ]
    
    cols = st.columns(4)
    for i, (emoji, emotion) in enumerate(emotions_display):
        with cols[i % 4]:
            st.markdown(f"**{emoji}** {emotion}")

if __name__ == "__main__":
    main()
