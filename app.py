import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf


model = tf.keras.models.load_model(
    'final_emotion_model_1.keras',
    custom_objects={'AttentionPooling': AttentionPooling},
    compile=False  # safer for inference
)

from tensorflow.keras import layers

class AttentionPooling(layers.Layer):
    """Attention-based pooling layer for better feature aggregation"""
    def __init__(self, units=64, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionPooling, self).build(input_shape)
    
    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=1)  # (batch_size, 1, features)
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)  # (batch_size, 1, units)
        ait = tf.tensordot(uit, self.u, axes=1)  # (batch_size, 1)
        ait = tf.nn.softmax(ait, axis=1)  # (batch_size, 1)
        weighted_input = x * tf.expand_dims(ait, axis=-1)  # (batch_size, 1, features)
        return tf.reduce_sum(weighted_input, axis=1)  # (batch_size, features)
    
    def get_config(self):
        config = super(AttentionPooling, self).get_config()
        config.update({'units': self.units})
        return config




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
