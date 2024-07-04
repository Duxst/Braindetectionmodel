Project Overview
This project aims to develop and implement a Convolutional Neural Network (CNN) for detecting brain tumors from MRI images. The model classifies images into two categories: tumorous and non-tumorous. The project leverages TensorFlow for building the CNN, OpenCV for image preprocessing, and scikit-learn for evaluating the model's performance

Dataset
The dataset consists of MRI images categorized into two folders: "yes" for tumorous images and "no" for non-tumorous images. These images are further split into training, validation, and test sets.

Data Preprocessing
The preprocessing steps include:

Loading images from directories
Resizing images to a standard size (240x240 pixels)
Normalizing pixel values to the range [0, 1]
Data augmentation to increase dataset diversity
Model Architecture
The CNN model consists of the following layers:

Input layer (240x240x3)
Three convolutional stages:
Conv2D -> BatchNormalization -> ReLU -> MaxPooling2D
Fully connected layers:
Flatten -> Dense (128 units, ReLU) -> Dense (1 unit, Sigmoid)
Training
The model is compiled with the Adam optimizer and binary cross-entropy loss function. It is trained for 25 epochs with a batch size of 32. The best model is saved based on validation accuracy using ModelCheckpoint, and training progress is logged with TensorBoard.

Evaluation
The model is evaluated on a separate test set using the following metrics:

Accuracy
Precision
Recall
F1 Score
ROC-AUC Score
Confusion Matrix and ROC Curve
The performance is visualized using confusion matrices and ROC curves to understand the model's behavior on different classes.

Results
Accuracy: 91%
Precision: 85%
Recall: 91%
F1 Score: 88%
ROC-AUC Score: 91%
