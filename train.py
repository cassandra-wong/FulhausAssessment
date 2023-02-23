import os
import cv2
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16

# Create a dataframe of image path and its label
img_dir = '/Users/cassandra/Desktop/FulhausAssessment/FulhausData'
furniture_label = ['Bed','Chair','Sofa']

def img_label_df(dir):
    img_list = []
    label_list = []
    for label in furniture_label:
        for img_file in os.listdir(dir+'/'+label):
            img_list.append(dir+'/'+label+'/'+img_file)
            label_list.append(label)
            
    df = pd.DataFrame({'img':img_list, 'label':label_list})
    return df

df = img_label_df(img_dir)

# Encode label
df_labels = {
    'Bed' : 0,
    'Chair' : 1,
    'Sofa' : 2
}

df['encode_label'] = df['label'].map(df_labels)

# Define image augmentation parameters
rotation_range = 20 
zoom_range = 0.1
horizontal_flip = True 

X = []

for img in df['img']:
    img = cv2.imread(str(img))
    rows,cols,channels = img.shape
    # augment_function(img)
    # random rotation 
    angle = np.random.uniform(-rotation_range, rotation_range)
    R = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img = cv2.warpAffine(img, R, (cols,rows))
    # random zoom (0.1)
    zoom = np.random.uniform(1-zoom_range, 1+zoom_range)
    pts1 = np.float32([[cols/2-rows/2*zoom,rows/2-rows/2*zoom],
                       [cols/2+rows/2*zoom,rows/2-rows/2*zoom],
                       [cols/2-rows/2*zoom,rows/2+rows/2*zoom]])
    pts2 = np.float32([[0,0],[cols,0],[0,rows]])
    Z = cv2.getAffineTransform(pts1,pts2)
    img = cv2.warpAffine(img, Z,(cols,rows))
    # random flip
    if horizontal_flip and np.random.random() < 0.5:
        img = cv2.flip(img, 1)
    # resize to 128 x 128 x 3
    img = cv2.resize(img, (128, 128))
    img = img/255
    X.append(img)

y = df['encode_label']

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val)

base_model = VGG16(input_shape=(128,128,3), include_top=False, weights='imagenet')

print(base_model.summary())

for layer in base_model.layers:
    layer.trainable = False

base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True

model = Sequential()
model.add(keras.Input(shape=(128,128,3)))
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(furniture_label), activation='softmax'))
model.summary()

model.compile(
  optimizer="adam",
  loss='sparse_categorical_crossentropy',
  metrics=['acc'])
history = model.fit(np.array(X_train), np.array(y_train), epochs=20, validation_data=(np.array(X_val), np.array(y_val)))

print(model.evaluate(np.array(X_test),np.array(y_test)))

plt.plot(history.history['acc'], marker='o')
plt.plot(history.history['val_acc'], marker='o')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.savefig('Accuracy.png')

plt.plot(history.history['loss'], marker='o')
plt.plot(history.history['val_loss'], marker='o')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('Loss.png')

model.save('vgg16_model.h5')

