import io
import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, jsonify, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import decode_predictions


# load the pre-trained VGG16 model
model = load_model('vgg16_model.h5')

# create a Flask app instance
app = Flask(__name__)

# preprocess image
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))
    img = img_to_array(img)
    img = img/255
    return img

# render html file
@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')

# define a route for the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # get the image data from the request
    image = request.files['image'].read()

    # preprocess the image data 
    preprocessed_image = preprocess_image(image)

    # make a prediction
    preds = model.predict(preprocessed_image)

    # return the prediction results
    results = decode_predictions(preds, top=3)[0]
    predictions = []
    for result in results:
        label = result[1]
        probability = float(result[2])
        predictions.append({'label': label, 'probability': probability})
        print('Predicted Class:', predictions[-1]['label'], 'Probability:', predictions[-1]['probability'])
    
    return render_template('index.html', pred=predictions[-1]['label'], prob=predictions[-1]['probability'])
    #return jsonify(predictions)


# start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
