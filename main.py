#Math and neural network imports
import cv2
import numpy as np
import os
import tensorflow as tf
import math

#GUI Imports
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel)
from PyQt6.QtCore import Qt, QPropertyAnimation, QRect, QEasingCurve
from PyQt6.QtGui import QFont, QPalette, QColor, QLinearGradient, QGradient

from aiPyQtTest import generateGUT;

#importing functions from aiGeneratedFunctions.py
from aiGeneratedFunctions import load_training_data;
from aiGeneratedFunctions import load_nonlabeled_data;
from aiGeneratedFunctions import print_image_array;
from aiGeneratedFunctions import display_predictions;

idToName = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i',
    19: 'j', 20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r',
    28: 's', 29: 't', 30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z',
    36: 'ampersand', 37: 'asterisk', 38: 'at', 39: 'dollar', 40: 'exclmark',
    41: 'greaterthan', 42: 'hash', 43: 'lessthan', 44: 'minus', 45: 'percent',
    46: 'plus', 47: 'quesmark'
}


#generateGUT();

def trainNeuralNetwork():

    #Getting standadrd training data
    '''
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    '''
    #Loading raw label and image data from file
    x_data, y_data = load_training_data("__Image Files__")
    
    print_image_array(x_data[0])
    print(y_data);

    x_train = x_data
    y_train = y_data

    print_image_array(x_train[len(x_train)-1])


    x_train = tf.keras.utils.normalize(x_train, axis=1)
    #x_test = tf.keras.utils.normalize(x_test, axis=1)


    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(28,28,1)));
    model.add(tf.keras.layers.Flatten());
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(70, activation='relu'))
    model.add(tf.keras.layers.Dense(48,activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x = x_train,y = y_train, epochs=10)

    model.save("JMEodel.keras")




def evaluateUsingLabeledData():
    model = tf.keras.models.load_model("JMEodel.keras")
    imagesToPredict = load_nonlabeled_data("28by28_Drawn");
    display_predictions(model,imagesToPredict,idToName)
    '''
    for image in imagesToPredict:
        try:
            print_image_array(image)
            prediction = model.predict(image)
            print(idToName[(int(str(np.argmax(prediction))))])
            plt.imshow(image[0], cmap=plt.cm.binary)
            plt.show()
        except:
            print("SOMETHING WENT WRONG")
    '''




def predictUsingModel():
    print("predictUsingModel");

while True:
    mode = input("MODE (Train or Evaluate or Predict or Stop) (T or E or P or S): ");
    if (mode == "T"):
        trainNeuralNetwork();
    elif (mode == "E"):
        evaluateUsingLabeledData();
    elif (mode == "P"):
        predictUsingModel();
    elif (mode == "S"):
        print("Bye");
        break;







