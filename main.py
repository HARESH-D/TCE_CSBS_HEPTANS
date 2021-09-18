import glob
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import os
import json
import numpy as np
from keras.models import model_from_json
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

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


X = []
y = []


for feature, label in train:
  X.append(feature)
  y.append(label)

test_data = list()
for img_path in glob.glob("E:\dataset\TTest\*.jpg"):
    try:
        img_arr = cv2.imread(img_path)[..., ::-1]  # convert BGR to RGB format
        resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
        test_data.append(resized_arr)
    except Exception as e:
        print(e)


# Normalize the data
X = np.array(X) / 255
test_data = np.array(test_data) / 255

X.reshape(-1, img_size, img_size, 1)
y = np.array(y)

test_data.reshape(-1, img_size, img_size, 1)

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    vertical_flip=True,
    horizontal_flip=True,
    validation_split=0.2,
    rotation_range=40 )

datagen.fit(X_train)

def create_model():
    model = Sequential()
    model.add(Conv2D(input_shape=(64, 64, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    model.summary()
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


opt = Adam(lr=0.0001)
model = create_model()

history = model.fit(datagen.flow(X_train, y_train, batch_size=32,
         subset='training'),
         validation_data=datagen.flow(X_train, y_train,
         batch_size=64, subset='validation'),
         epochs=50)


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
#
#
# load json and create model
json_file = open('backup_models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
#
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])
predictions = loaded_model.predict_classes(test_data).tolist()
print(predictions)
# print(y_test)


# from utils.io import write_json
def write_json(filename, result):
    with open(filename, 'w') as outfile:
        json.dump(result, outfile)


def read_json(filename):
    with open(filename, 'r') as outfile:
        data =  json.load(outfile)
    return data


def generate_sample_file(filename, x):
    res = {}
    for i in range(1,98):
        test_set = str(i) + '.jpg'
        res[test_set] = x[i]
    print("file generate")
    write_json(filename, res)


if __name__ == '__main__':
    generate_sample_file('./sample_result1.json', predictions)
