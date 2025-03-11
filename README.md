# GAN for MNIST Digits Generation

This project implements a Generative Adversarial Network (GAN) to generate handwritten digits similar to the MNIST dataset. The model consists of a Generator that produces synthetic images and a Discriminator that distinguishes between real and fake images.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Training the GAN](#training-the-gan)
- [Generating Images](#generating-images)
- [Results](#results)
- [License](#license)

## Introduction
Generative Adversarial Networks (GANs) are a type of deep learning model consisting of two neural networks that compete against each other:
- **Generator**: Learns to generate realistic images from random noise.
- **Discriminator**: Learns to distinguish between real and fake images.

The two networks are trained together in a game-theoretic framework to improve the quality of generated images over time.

## Dataset
The GAN is trained on the MNIST dataset, which consists of 60,000 training images of handwritten digits (0-9), each of size 28x28 pixels.

## Model Architecture
### Generator
- Fully connected layers progressively increase dimensionality from random noise to a 28x28 image.
- Uses ReLU activation and Tanh activation for output.

### Discriminator
- Fully connected layers classify images as real or fake.
- Uses LeakyReLU activation and Sigmoid activation for output.

## Installation
To run this project, install the required dependencies:
```sh
pip install torch torchvision matplotlib numpy
```

## Training the GAN
Run the following command to train the GAN:
```sh
python train_gan.py
```
This will:
- Load the MNIST dataset
- Train the Generator and Discriminator
- Save the trained generator model as `generator.pth`

## Generating Images
Once training is complete, use the trained model to generate new handwritten digits:
```sh
python generate.py
```
This script loads the trained generator and generates a set of fake MNIST images.

## Results
After training, the generator is capable of producing realistic-looking handwritten digits. The images below show some examples of generated digits:


![Fig1](https://github.com/user-attachments/assets/41e821cd-4ce7-472d-8883-4b730ea16269)

## License
This project is open-source and available under the MIT License.

