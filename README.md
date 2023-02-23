# Furniture Classification Take Home Assessment

## Overview

In this project, image classification is performed using VGG16 to classify 3 furniture types - bed, chair, and sofa. Each class contains 100 images. A small convolutional neural network (CNN) was created as baseline, and used VGG16 through transfer learning to improve results. See `FulhausAssessment.ipynb` for exploratory data analysis, visualization, and model building. The training app is saved inside `train.py`. 

## Requirements

- JupyterNotebook or Google Colab
- Python3
- Pip

## Installation of the API / Web Application

1. Create and activate virtual environment.

```sh
$ python -m venv python3-virtualenv
$ source python3-virtualenv/bin/activate
```

2. Clone the git repository.

```sh
git clone https://github.com/cassandra-wong/FurnitureClassification.git
```

```bash
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
