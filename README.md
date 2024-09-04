This repository contains the Python code for diagnosing epileptic seizures using deep learning techniques, specifically Convolutional Neural Networks (CNN) and 
Bidirectional Long Short-Term Memory (BiLSTM) networks. The original ensemble learning approach has been removed, and this version focuses solely on deep learning methods to classify EEG signals.

# Table of Contents:

1- Installation

2- Data Preprocessing

3- Model Architectures

4- Training

5- Evaluation

6- Results


# Installation
Clone the repository:

    git clone https://github.com/danielnrz/epileptic-seizure-diagnosis.git
    cd epileptic-seizure-diagnosis

Install dependencies:
        
    pip install -r requirements.txt


Download the dataset and place it in the data/ directory.

# Data Preprocessing

The data preprocessing involves several steps to prepare the EEG data for input into the models:

Load the Dataset:

The EEG data is loaded from a CSV file located at '/content/drive/MyDrive/Colab Notebooks/DataSets/Epileptic Seizure Recognition.csv'.


Data Preparation:

The dataset is divided into different categories based on seizure types.
Data is split into training, validation, and test sets using random sampling.
Reshape Data:

The data is reshaped by adding an extra dimension, making it suitable for processing by CNN and BiLSTM models.
This preprocessing is handled in the initial part of the code.

# Model Architectures
Convolutional Neural Network (CNN)
The CNN architecture extracts spatial features from the EEG signals:

Input Shape: The input is a 1D signal reshaped with an additional dimension.
Layers:
Conv1D Layers: Two convolutional layers with 32 and 64 filters, respectively.
MaxPooling: Applied after each convolutional layer to reduce the dimensionality.
Flatten: Converts the 3D output into a 1D feature vector.
Dense Layers: Fully connected layers to learn complex patterns.
Output Layer: Final dense layer with 2 units for binary classification (epileptic vs. non-epileptic).
Bidirectional LSTM (BiLSTM)
The BiLSTM model captures temporal dependencies in the EEG signals:

Input Shape: Similar to the CNN, the input is a reshaped 1D signal.
Layers:
Dense Layer: First dense layer with ReLU activation.
Bidirectional LSTM: Processes the data in both forward and backward directions, with 128 units.
Dropout & Batch Normalization: Used to prevent overfitting and stabilize training.
Output Layer: Dense layer with softmax activation for classification.
Both model architectures are defined in network_CNN() and network_LSTM() functions, respectively.

# Training
The models are trained on the preprocessed EEG data with the following configuration:

Optimizer: Adam optimizer.
Loss Function: Sparse categorical crossentropy for classification.
Metrics: Accuracy is tracked during training.
Epochs: Each model is trained for 150 epochs.
Batch Size: 32 samples per batch.
The training process involves monitoring validation accuracy, with the best model weights saved using Keras callbacks (ModelCheckpoint).

Training CNN Model
The CNN model is trained on both "epileptic vs non-epileptic" data and "epileptic vs all" data.

#### python

    # Training CNN model on epileptic vs all data
    history3 = model3.fit(data_train_all, label_train_all, epochs=150, batch_size=32, validation_data=(data_val_all, label_val_all), callbacks=[model_checkpoint_callback])
    evaluate_model(history3, data_test_all, label_test_all, model3)
## Training BiLSTM Model
The BiLSTM model is also trained on the "epileptic vs non-epileptic" data, following a similar training process as the CNN model.

#### python
    # Training BiLSTM model on epileptic vs all data
    history4 = model4.fit(data_train_all, label_train_all, epochs=150, batch_size=32, validation_data=(data_val_all, label_val_all), callbacks=[model_checkpoint_callback])
    evaluate_model(history4, data_test_all, label_test_all, model4)
# Evaluation
Model performance is evaluated using the test set, with the following outputs:

Accuracy and Loss Plots: Visualize the model’s performance over epochs.
Confusion Matrix: A confusion matrix is generated to assess classification accuracy across different classes.
The evaluate_model() function is used to perform the evaluation and generate plots.

Additionally, predictions are made using the trained BiLSTM model, and the accuracy, precision, recall, and F1 score are calculated:

#### python
    # Evaluate BiLSTM model predictions
    accBiLSTM = round(accuracy_score(y_test, predictionOfBiLSTM) * 100, 2)
    p = precision_score(y_test, predictionOfBiLSTM)
    r = recall_score(y_test, predictionOfBiLSTM)
    f1 = f1_score(y_test, predictionOfBiLSTM)
# Results
The models achieve the following results on the test set:

CNN Model (Epileptic vs Non-Epileptic): Accuracy, Precision, Recall, and F1-Score.
CNN Model (Epileptic vs All): Performance metrics for a more challenging multi-class classification task.
BiLSTM Model: Similar metrics, providing a comparison between CNN and LSTM-based approaches.
