## Overview
Human emotion is a complex psychological state that involves a variety of feelings, thoughts, and behaviors. Emotions are often associated with physiological changes and reactions to different stimuli. They play a crucial role in human behavior and interaction, influencing how individuals respond to their environment, make decisions, and communicate with others.

Computer science has significantly impacted the understanding, recognition, and interaction with human emotions through various technologies and applications. 
- It will help to understand customer emotions and intentions will improve marketing strategies.
- Virtual Assistant can interact with user appropriately and enhance human-computer interaction.
- Facial Expression Analysis and Voice analysis helps to understand user mental state and hence improve virtual mental health Apps.
  
## Project Description
The project involves building a real-time emotion detection system that can analyze facial expressions and determine the corresponding emotions from live video. By utilizing MobileNet for feature extraction and OpenCV for image processing, the system is designed to process video feeds and detect emotions in real-time.

## What is done in this project
![image](https://github.com/user-attachments/assets/e41839e5-8f1d-46e9-9591-5ccfdbc00d61)

1.) Data Collection and Preprocessing: Acquired and preprocessed the dataset of facial expressions.<br>
2.) Model Training: Trained a MobileNet-based deep learning model to recognize different emotions and stored the model in H5 file as model.h5.<br>
3.) Integration with OpenCV: Integrated the trained model with OpenCV for real-time video processing.<br>
4.) Real-time Emotion Detection: Implemented a system to capture video from a webcam, process each frame, and display the detected emotion in real-time.

## TechStack and frameworks Requirements
- Programming Language: Python
- Frameworks and Libraries: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Scikit-learn
- Development Environment: Jupyter Notebook, Anaconda

## Dataset Description
For this project i utilized this dataset from [FER-2013 Dataset on Kaggle](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)The dataset consists of images of human faces labeled with various emotions. Our dataset consists of two folders, one for training purposes, which will be trained repeatedly for 32 times as we take batch size 32 and the other folder for testing the model, which again with batch size 32. Each folder consists of 7 folders with a collection of different emotions images . These sets of emotions are happy, sad, fear, neutral, anger, surprise, disgust. Training datasets consist of 28709 images while testing datasets include 7178 images.Total images files are approximately 35887 images.

  ![image](https://github.com/user-attachments/assets/1b3425d9-e105-4dd8-959f-5ecc70b1a68f)

- Classes: The dataset includes seven emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
- Image Size: All images are resized to 224x224 pixels to match the input size of MobileNet.
- Preprocessing Steps: The images are normalized to a range of [0, 1] and augmented using zoom, shear, and horizontal flip transformations to increase the diversity of the training data.

## Image Preprocessing
The image preprocessing techniques used in your project include rescaling, resizing, grayscale conversion, and various data augmentation methods such as rotation, shearing, zooming, width and height shifting, and horizontal flipping.
- Rescaling: Normalizes pixel values to a common range, typically between 0 and 1.
- Resizing: Ensures that all input images have the same dimensions, which is necessary for feeding the images into the model.
- Grayscale Conversion: Converts RGB images to grayscale, reducing the complexity and number of channels from 3 to 1.
- Data Augmentation: Increases the diversity of the training dataset by applying random transformations to the input images, which helps in reducing overfitting and improving the model's robustness.
-- Rotation: Rotates the images randomly within a specified range.
-- Shear: Applies random shearing transformations.
-- Zoom: Randomly zooms into images.
-- Width Shift: Randomly shifts images horizontally.
-- Height Shift: Randomly shifts images vertically.
-- Horizontal Flip: Randomly flips images horizontally.
-- Fill Mode: Defines how to fill in new pixels that may appear after a transformation.
  
## What is MobileNet?
MobileNet is a lightweight, efficient convolutional neural network architecture designed for mobile and embedded vision applications. It uses depthwise separable convolutions to significantly reduce the number of parameters and computational cost.

![flowchart (1)](https://github.com/user-attachments/assets/7db71595-c49f-4fc0-a7a3-bc5af7ea901a)

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

![image](https://github.com/user-attachments/assets/1a15fbb2-ac10-4fc7-8b39-5e608a4361a3)

![image](https://github.com/user-attachments/assets/ff4f06e9-22a7-4189-a271-42b277618125)

![image](https://github.com/user-attachments/assets/41eaff48-14a5-4a92-b858-abcdc32aa47c)


## Results
