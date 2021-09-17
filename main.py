import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam


import tensorflow as tf

import cv2
import os

import numpy as np

labels = ['hi', 'background']
img_size = 64
def get_data(data_dir):
    data = []
    for label in labels:

        #path. join() method in Python join one or more path components intelligently
        path = os.path.join(data_dir, label)

        #index() is an inbuilt function in Python, which searches for a given element
        #from the start of the list and returns the lowest index where the element appears.
        class_num = labels.index(label)

        #listdir() method in python is used to get the list of all files and directories in the specified directory.
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

train = get_data(r"E:\dataset\training")
val = get_data(r"E:\dataset\test")

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

model = Sequential()
model.add(Conv2D(input_shape=(64,64,3),filters=64,kernel_size=(3,3),padding="same", activation="tanh"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="tanh"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="tanh"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="tanh"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="tanh"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="tanh"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="tanh"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="tanh"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="tanh"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="tanh"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="tanh"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="tanh"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="tanh"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="tanh"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="tanh"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))


model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

opt = Adam(lr=0.000001)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

history = model.fit(x_train,y_train,epochs = 100 )

predictions = model.predict_classes(x_val)
print(predictions)
print(y_val)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

from keras.models import model_from_json
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(x_train, y_train, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
