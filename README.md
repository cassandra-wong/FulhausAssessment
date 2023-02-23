# Furniture Classification Take Home Assessment

## Overview

In this project, image classification is performed using VGG16 to classify 3 furniture types - bed, chair, and sofa. Each class contains 100 images. A small convolutional neural network (CNN) was created as baseline, and used VGG16 through transfer learning to improve results. The following steps are implemented to achieve this goal:

- Build a classification model using deep learning model
  - `FulhausAssessment.ipynb` and `train.py`
- Build an API to access the model using Flask
  - `app.py`
- Create a Docker image of your code by following docker best practices
  - `Dockerfile`
- Implement CI/CD pipeline on Github Actions
- Add a clear README file with instructions

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
