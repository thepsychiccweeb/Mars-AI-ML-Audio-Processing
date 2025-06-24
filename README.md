# Mars-AI-ML-Audio-Processing
# Audio Emotion Recognition using Deep Learning

A comprehensive emotion recognition system that analyzes audio files to detect human emotions using machine learning techniques trained on the RAVDESS dataset.

## Overview
This project implements an end-to-end pipeline for recognizing emotions from speech and song audio using TensorFlow and advanced feature extraction techniques. The system can classify audio into 8 different emotional states with high accuracy using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

## Dataset
- **RAVDESS Dataset**: Contains speech and song recordings from 24 professional actors
- **8 Emotions**: Happy, Sad, Angry, Fearful, Surprised, Disgust, Neutral, Calm
- **Audio Format**: High-quality WAV files with controlled recording conditions

## Model Architecture
### ResNet Implementation
Our custom ResNet architecture includes:
- **Input Layer**: Accepts 45 extracted audio features
- **Residual Blocks**: 3 blocks with skip connections to prevent vanishing gradients
- **Dense Layers**: 128, 64, 32 neurons with ReLU activation and batch normalization
- **Skip Connections**: Add input to output of each block, enabling deeper training
- **Regularization**: Dropout (0.3-0.5) and L2 regularization to prevent overfitting
- **Output Layer**: 8 neurons with softmax activation for emotion classification

The ResNet architecture allows for deeper networks while maintaining gradient flow through skip connections, crucial for learning complex emotion patterns from audio features.

## Features
- **Audio Processing**: Extracts 45 key features including MFCC, spectral features, chroma, and contrast
- **Multiple Models**: Implements MLP and ResNet architectures for comparison
- **Feature Selection**: Uses SelectKBest for optimal feature selection
- **Cross-Validation**: Comprehensive model evaluation with k-fold validation
- **Interactive App**: Streamlit web application for real-time emotion prediction

## Pipeline Steps
1. **Data Loading**: Process RAVDESS WAV audio files with emotion labels
2. **Feature Extraction**: Extract 45 audio features (MFCC, ZCR, RMS, spectral features)
3. **Preprocessing**: Apply feature selection and standardization
4. **Model Training**: Train neural networks with regularization and callbacks
5. **Evaluation**: Cross-validation, confusion matrices, and performance analysis
6. **Deployment**: Streamlit app for interactive emotion prediction

## Results
- **Final Test Accuracy**: 84.6% on RAVDESS dataset
- **Confusion matrix and other important results are in result_screenshots folder**
- **Best Model**: ResNet architecture with enhanced feature engineering
- **Generalization Gap**: <10% indicating good model generalization
- **Cross-Validation**: Consistent performance across multiple folds
- **Confusion Matrix**: Strong performance on distinct emotions (angry, happy, sad) with some confusion between similar emotions (calm/neutral)

## Supported Emotions
Happy, Sad, Angry, Fearful, Surprised, Disgust, Neutral, Calm

The system demonstrates robust emotion recognition capabilities suitable for real-world applications in speech emotion analysis.

## Supported Emotions
Happy, Sad, Angry, Fearful, Surprised, Disgust, Neutral, Calm

The system demonstrates robust emotion recognition capabilities suitable for real-world applications in speech emotion analysis.ch emotion analysis.

#Dataset- https://zenodo.org/records/1188976#.XCx-tc9KhQI
#Streamlit app- https://mars-ai-ml-audio-processing-bfwfa3talck68n6dr8uju8.streamlit.app/
   (or you can also use app.py to deploy it using streamlit)
