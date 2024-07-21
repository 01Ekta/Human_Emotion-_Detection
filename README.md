## Overview

## Project Description

## What is done in this project

## TechStack and frameworks Requirements
- Programming Language: Python
- Frameworks and Libraries: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Scikit-learn
- Development Environment: Jupyter Notebook, Anaconda

## Dataset Description
For this project i utilized this dataset from [FER-2013 Dataset on Kaggle]([https://www.dropbox.com/s/nilt43hyl1dx82k/dataset.zip?dl=0](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer))The dataset consists of images of human faces labeled with various emotions. The key attributes of the dataset include:

- Classes: The dataset includes seven emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
- Image Size: All images are resized to 224x224 pixels to match the input size of MobileNet.
- Preprocessing Steps: The images are normalized to a range of [0, 1] and augmented using zoom, shear, and horizontal flip transformations to increase the diversity of the training data.

## Image Preprocessing

## What is MobileNet?

## What is OpenCv?
OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It provides a comprehensive set of tools for image and video processing.

In this project, OpenCV is used for:
- Video Capture: Accessing the webcam to capture real-time video feeds.
- Face Detection: Detecting faces in video frames using Haar cascades.
- Image Processing: Preprocessing the detected faces before passing them to the MobileNet model for emotion detection.

## Real-Time Video emotion Detection

## Results
