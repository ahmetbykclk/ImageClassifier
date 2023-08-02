## Cats and Dogs Image Classifier

This repository contains a deep learning model trained to classify images as either cats or dogs. The model uses a pre-trained MobileNetV2 base with a custom classification layer added on top. The training and testing images are loaded using the ImageDataGenerator from Keras.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [How to Use](#how-to-use)
- [Model Architecture](#model-architecture)

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python (>= 3.6)
- TensorFlow (>= 2.x)
- Keras (>= 2.x)

Install the required packages using the following command:

pip install tensorflow keras

## Dataset

The dataset used for training and testing the model is not included in this repository. Please download the dataset and organize it into the following structure:

dataset

|-- training_set

|----|-- cats

|----|-- dogs

|-- test_set

|----|-- cats

|----|-- dogs

The training set contains subdirectories cats and dogs, each containing images of cats and dogs, respectively. The test set follows the same structure.

You can download my dataset directly from this link: 

https://www.kaggle.com/datasets/chetankv/dogs-cats-images/download?datasetVersionNumber=1.

## How to Use

1- Clone the repository to your local machine using the following command:

git clone https://github.com/your-username/cats-dogs-classifier.git

cd cats-dogs-classifier

2- Organize the dataset as described above.

3- Run the training script to train the model: 

python model.py

After training, the model will be saved as cats_dogs_classifier.h5.

4- To test the model's predictions on cat images, run the following:

python cattest.py

5- To test the model's predictions on dog images, run the following:

python dogtest.py

## Model Architecture


The model architecture consists of a MobileNetV2 base with the top classification layer replaced by a GlobalAveragePooling2D layer and a Dense layer with a sigmoid activation function. The base model's layers are frozen during training to prevent overfitting on the small dataset.

