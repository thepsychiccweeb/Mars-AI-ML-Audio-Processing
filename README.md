# Mars-AI-ML-Audio-Processing
Audio Emotion Recognition using Deep Learning
A comprehensive emotion recognition system that analyzes audio files to detect human emotions using machine learning techniques trained on the RAVDESS dataset.

Overview
This project implements an end-to-end pipeline for recognizing emotions from speech and song audio using TensorFlow and advanced feature extraction techniques. The system can classify audio into 8 different emotional states with high accuracy using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

Dataset
RAVDESS Dataset: Contains speech and song recordings from 24 professional actors
8 Emotions: Happy, Sad, Angry, Fearful, Surprised, Disgust, Neutral, Calm
Audio Format: High-quality WAV files with controlled recording conditions
Features
Audio Processing: Extracts 45 key features including MFCC, spectral features, chroma, and contrast
Multiple Models: Implements MLP and ResNet architectures for comparison
Feature Selection: Uses SelectKBest for optimal feature selection
Cross-Validation: Comprehensive model evaluation with k-fold validation
Interactive App: Streamlit web application for real-time emotion prediction
Pipeline Steps
Data Loading: Process RAVDESS WAV audio files with emotion labels
Feature Extraction: Extract 45 audio features (MFCC, ZCR, RMS, spectral features)
Preprocessing: Apply feature selection and standardization
Model Training: Train neural networks with regularization and callbacks
Evaluation: Cross-validation, confusion matrices, and performance analysis
Deployment: Streamlit app for interactive emotion prediction
Results
Final Test Accuracy: ~75-80% on RAVDESS dataset
Best Model: ResNet architecture with enhanced feature engineering
Generalization Gap: <10% indicating good model generalization
Cross-Validation: Consistent performance across multiple folds
Confusion Matrix: Strong performance on distinct emotions (angry, happy, sad) with some confusion between similar emotions (calm/neutral)
Supported Emotions
Happy, Sad, Angry, Fearful, Surprised, Disgust, Neutral, Calm

The system demonstrates robust emotion recognition capabilities suitable for real-world applications in speech emotion analysis.
