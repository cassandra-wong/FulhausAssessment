from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import cv2

# load the pre-trained VGG16 model
model = load_model('vgg16_model.h5')

# create a Flask app instance
app = Flask(__name__)

# preprocess image
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))
    img = img/255

# define a route for the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # get the image data from the request
    image = request.files['image'].read()

    # preprocess the image data (you will need to define this function)
    preprocessed_image = preprocess_image(image)

    # make a prediction
    prediction = model.predict(preprocessed_image)

    # return the prediction as a JSON response
    return jsonify(prediction.tolist())

# start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
