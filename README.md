# Plant Disease Detection using CNN

This project focuses on detecting plant diseases using Convolutional Neural Networks (CNNs). The model is trained on a dataset of plant images to classify various plant diseases effectively.

## Project Overview
- **Framework:** TensorFlow/Keras  
- **Dataset:** [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)  
- **Goal:** Identify plant diseases such as Tomato Early Blight and Apple Scab based on leaf images.

## Features
- Data preprocessing and augmentation.
- Custom CNN architecture with multiple convolutional and pooling layers.
- Dropout layers to prevent overfitting.
- Visualization of training and validation accuracy.
- Prediction and visualization of disease labels for test images.

## How to Use

### Training
1. Download the dataset from [here](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).  
2. Prepare the training and validation datasets in respective directories (`train` and `valid`).  
3. Run the training script to train the model:
    ```bash
    python train.py
    ```
4. The trained model will be saved as `trained_plant_disease_model.keras`.

### Testing
1. Run the testing script:
    ```bash
    python test.py
    ```
2. Provide the path to a test image to visualize and classify the disease.

## Results
- Training and validation accuracy visualized over epochs.
- Prediction of plant diseases from test images.

## Prerequisites
- Python 3.x  
- TensorFlow  
- NumPy  
- OpenCV  
- Matplotlib  
- Seaborn  
- Pandas  

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/aqib11234/plant-disease-detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd plant-disease-detection
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
