#Math and neural network imports
import cv2
import numpy as np
import os
import tensorflow as tf

#GUI Imports
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel)
from PyQt6.QtCore import Qt, QPropertyAnimation, QRect, QEasingCurve
from PyQt6.QtGui import QFont, QPalette, QColor, QLinearGradient, QGradient

from aiPyQtTest import generateGUT;

#from aiGeneratedFunctions import load_training_data; a

#generateGUT();

def trainNeuralNetwork():

    #Getting standadrd training data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)


    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(36,activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x = x_train,y = y_train, epochs=3)

    model.save("JMEodel.keras")

def evaluateUsingLabeledData():
    model = tf.keras.models.load_model("JMEodel.keras")
    image_number = 4
    while os.path.isfile(f"28by28_Drawn/digit{image_number}.png"):
        try:
            img = cv2.imread(f"28by28_Drawn/digit{image_number}.png")[:,:,0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print(f"This digit is probably a {np.argmax(prediction)}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except:
            print("error");
        finally:
            image_number += 1;

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








    
    #loss, accuracy = model.evaluate(x_test, y_test)

    #print(loss);
    #print(accuracy);