## Overview

## Project Description

## What is done in this project

## TechStack and frameworks Requirements
- Programming Language: Python
- Frameworks and Libraries: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Scikit-learn
- Development Environment: Jupyter Notebook, Anaconda

## Dataset Description
For this project i utilized this dataset from [FER-2013 Dataset on Kaggle](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)The dataset consists of images of human faces labeled with various emotions. The key attributes of the dataset include:

- Classes: The dataset includes seven emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
- Image Size: All images are resized to 224x224 pixels to match the input size of MobileNet.
- Preprocessing Steps: The images are normalized to a range of [0, 1] and augmented using zoom, shear, and horizontal flip transformations to increase the diversity of the training data.

## Image Preprocessing

## What is MobileNet?
MobileNet is a lightweight, efficient convolutional neural network architecture designed for mobile and embedded vision applications. It uses depthwise separable convolutions to significantly reduce the number of parameters and computational cost.

In this project, MobileNet is used as the base model for feature extraction. The top layers of MobileNet are replaced with custom layers tailored for emotion classification:

Base Model: MobileNet with the top layers removed.
Custom Layers: Flatten layer followed by a dense layer with seven output units (one for each emotion class) and a softmax activation function.

1.) Loading MobileNet:

The MobileNet model is loaded with pre-trained weights from ImageNet, excluding the top fully connected layers. This helps in leveraging pre-learned features for the new task of emotion detection. The input shape is set to (224, 224, 3).

2.) Freezing Base Model:
The base MobileNet layers are frozen (base_model.trainable = False) to retain the pre-trained features and prevent them from being updated during training. This is a standard practice in transfer learning to utilize existing knowledge.

3.) Adding Custom Layers:
- Flatten Layer: Converts the 2D output of the base model to a 1D tensor.
- Dense Layer (1024 units, ReLU activation): Adds a fully connected layer with 1024 neurons and ReLU activation function.
- Dropout Layer (0.5): Introduces dropout to reduce overfitting by randomly setting 50% of the input units to zero during training.
- Dense Layer (7 units, Softmax activation): Final output layer with 7 neurons (one for each emotion class) and softmax activation to classify the emotions.
  
4.) Compiling the Model:
The model is compiled using the Adam optimizer and categorical cross-entropy loss function, which is appropriate for multi-class classification problems.

5.) Data Generators:
ImageDataGenerator: This is used for real-time data augmentation, which involves applying random transformations (such as rotation, shear, zoom, and flip) to the training images. This helps in improving the model's generalization by providing more diverse training samples.

6.) Training the Model:
The model is trained using the fit method, with the training and validation data generators providing augmented data. This setup ensures that the model learns effectively from the available dataset and performs well on unseen data.

## What is OpenCv?
OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It provides a comprehensive set of tools for image and video processing.

In this project, OpenCV is used for:
- Video Capture: Accessing the webcam to capture real-time video feeds.
- Face Detection: Detecting faces in video frames using Haar cascades.
- Image Processing: Preprocessing the detected faces before passing them to the MobileNet model for emotion detection.

## Real-Time Video emotion Detection

## Results
