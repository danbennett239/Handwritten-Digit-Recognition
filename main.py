import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf


# First train a model, then enable the 'use_model' or 'use_model_convolutional' Depending on the neural network trained.
# Store user created digits within the 'digits' folder - make sure they are 28x28 px and following naming convention.
# Enable 'run_test'.

train_new_model = False
use_model = False
train_new_model_convolutional = False
use_model_convolutional = False
run_test = False

if train_new_model:
    # Import dataset from tf
    mnist = tf.keras.datasets.mnist

    #Split into testing and training data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #Normalise data
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    #Neural network
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    #Output layer - 10 digits, 10 neurons
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)

    model.save('handwritten.model')

if use_model:
    model = tf.keras.models.load_model('handwritten.model')

if train_new_model_convolutional:
    # Import dataset from tf
    mnist = tf.keras.datasets.mnist

    # Split into testing and training data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize data
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # Reshape data to have a channel dimension
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Convolutional Neural Network
    model = tf.keras.models.Sequential()

    # Convolutional layers
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())

    # Fully connected layers
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)

    model.save('handwritten_cnn.model')

if use_model_convolutional:
    model = tf.keras.models.load_model('handwritten_cnn.model')



if run_test:
    image_number = 1
    while os.path.isfile('digits/digit{}.png'.format(image_number)):
        try:
            img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print("The number is probably a {}".format(np.argmax(prediction)))
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
            image_number += 1
        except:
            print("Error reading image! Proceeding with next image...")
            image_number += 1


