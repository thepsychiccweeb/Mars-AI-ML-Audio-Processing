import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf

# Load your trained model
from my_custom_layers import AttentionPooling  # or define it inline

model = tf.keras.models.load_model('final_emotion_model_1.keras', custom_objects={'AttentionPooling': AttentionPooling})



# Define label map (change as per your dataset)
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Preprocessing function
def extract_features(audio_path, max_len=174):
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # Padding or truncating to fixed length
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Add batch & channel dims
    return mfcc

# Streamlit UI
st.title("🎤 Emotion Recognition from Voice")
st.markdown("Upload a `.wav` file to predict the emotion.")

uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio("temp.wav", format='audio/wav')
    
    st.write("🔄 Extracting features...")
    features = extract_features("temp.wav")

    st.write("🧠 Making prediction...")
    prediction = model.predict(features)
    predicted_label = emotion_labels[np.argmax(prediction)]

    st.success(f"🎯 Predicted Emotion: **{predicted_label}**")
