# Emotion Classification with Bi-LSTM

## Overview
This repository contains a machine learning project for classifying emotions from text data using a Bidirectional Long Short-Term Memory (Bi-LSTM) neural network. The project processes text datasets (train, validation, and test sets) to predict emotions such as anger, fear, joy, love, sadness, and surprise.

## Project Structure
- `train.csv`, `val.csv`, `test.csv`: Processed datasets containing text and corresponding emotion labels.
- `emotion_classifier.h5`: The trained Bi-LSTM model saved in HDF5 format.
- `preprocessing.py`: Python script for data preprocessing (cleaning, tokenization, padding).
- `model_training.py`: Python script for building, training, and evaluating the Bi-LSTM model.
- `README.md`: This file, providing project documentation.

## Requirements
To run this project, you need the following dependencies:
- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn

Install the dependencies using:
```bash
pip install tensorflow numpy pandas nltk scikit-learn matplotlib seaborn
