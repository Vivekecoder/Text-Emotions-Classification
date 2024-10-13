## Overview
The Emotion Classification Model is a machine learning project that aims to classify emotions expressed in text. Using a dataset containing sentences labeled with their corresponding emotions, this project leverages the Keras and TensorFlow libraries to build, train, and evaluate a neural network model capable of predicting emotions from user input. 

This model is particularly useful for applications in sentiment analysis, customer feedback analysis, and social media monitoring, where understanding user sentiment is crucial.

## Features
- **Text Preprocessing**: The input text is cleaned to remove special characters and convert all text to lowercase, enhancing the quality of the training data.
- **Tokenization**: Utilizes Keras' `Tokenizer` to convert text data into sequences of integers, representing the words in the dataset.
- **Padding Sequences**: Ensures that all input sequences are of uniform length, which is necessary for training the neural network.
- **Neural Network Architecture**: Implements a simple yet effective feedforward neural network using embedding layers for text representation.
- **User Input for Prediction**: Allows users to input sentences directly, enabling real-time emotion predictions based on the trained model.
- **Evaluation Metrics**: Provides classification reports and confusion matrices to evaluate model performance comprehensively.


## Install dependencies
Create a requirements.txt file or install the necessary libraries directly:
pip install pandas numpy keras tensorflow seaborn matplotlib scikit-learn

## Usage
Prepare your dataset in a text file (train.txt) formatted with two columns: Text (the input sentence) and Emotions (the corresponding emotion label), separated by a semicolon.
Run the main script to train the model
After training, you can input sentences to predict their emotions interactively.
After training the model, the performance is evaluated using classification metrics and a confusion matrix. The classification report includes precision, recall, and F1-score for each emotion class, giving insights into how well the model performs.
