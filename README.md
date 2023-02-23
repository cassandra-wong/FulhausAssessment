# Furniture Classification Take Home Assessment

## Overview

In this project, image classification is performed using VGG16 to classify 3 furniture types - bed, chair, and sofa. Each class contains 100 images. VGG16 and transfer learning is selected as the architecture model. The following steps are implemented to achieve this goal:

- Build a classification model using deep learning model
  - `FulhausAssessment.ipynb` and `train.py`
- Build an API to access the model using Flask
  - `app.py`
- Create a Docker image of your code by following docker best practices
  - `Dockerfile`
- Implement CI/CD pipeline on Github Actions
- Add a clear README file with instructions

## Model

### Architecture

VGG-16 is a convolutional neural network (CNN) that is 16 layers deep. It is selected as the backbone of the classifier as it is one of the popular algorithms for image classification and is easy to use it for transfer learning. 

Firstly, the VGG16 model parameters are freezed, and only parameters in the last layer are allowed to be adjusted to reduce computation workload. Then, custom layers are added to the base model -- Flatten, Dropout, and Dense. The last layer has an output shape equal to 3, the output will be probabilities of 3 furniture categories, and the softmax function will summarize the category with the highest probability.

### Data Preprocessing

1. Encode label

    The labels ['bed','chair','sofa'] are encoded from a categorical to a numerical value for deep learning purposes.

2. Image sizing

    Before an image is inputted for deep learning, it must be converted to an array with a specific size (in this case, 128x128) for the model and scaled to have a value between 0 and 1.

3. Image Augmentation

    Due to the small set of samples, image augmentation is implemented to apply random perturbations that preserve the label information. Examples of transformations include horizontal blurring, rotations, and zoom.

### Training

We split the dataset into three folds: training (80%), test (16%) and validation (16%), and use the `sparse_categorical_crossentropy` as our target along with the `accuracy` as the evaluation metric. The train dataset obtained an accuracy of 99.11%, with a loss of 0.0152, and the test dataset obtained an accuracy of 85.71%, with a loss of 0.5983. 

## Requirements

- JupyterNotebook or Google Colab
- Python 3.7-3.10
- Pip 20.3 or higher

## Installation of the API / Web Application

1. Create and activate virtual environment.

```sh
$ python3 -m venv python3-virtualenv
$ source python3-virtualenv/bin/activate
```

2. Clone the git repository and enter folder.

```sh
git clone https://github.com/cassandra-wong/FurnitureClassification.git
```
```sh
cd FurnitureClassification
```
```sh
pip install -r requirements.txt 
```

3. The method to load the application is dependent on the operating system.

For Unix Bash (Linux, Mac, etc.) Users:
```sh
$ export FLASK_APP=app.py 
$ flask run
```

For Windows CMD:
```sh
> set FLASK_APP=app.py 
> flask run
```

4. Click the link where the app is hosted on localhost.


## Docker Image

1. Build a Docker image.

```sh
docker build -t [NAME] .
```

2. Run the Docker image.
```sh
docker run -p 5000:5000 [NAME]
```

## Conclusion

This project provided me with an opportunity to build a deep learning model to classify furniture types using tensorflow, build an API to access the model with Flask, create a Docker image, as well as implemented CI/CD pipeline on Github Actions. 

Given more time, I would test out different architectures (i.e., ResNet, MobileNetV2, etc.) and tune the hyperparameters, as well as create a more aesthetically looking UI for the web application. 
