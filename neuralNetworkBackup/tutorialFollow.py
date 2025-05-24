import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

mode = input("MODE (Train or Evaluate) (T or E): ")

def trainNeuralNetwork():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x = x_train,y = y_train, epochs=3)

    model.save("C:/Users/joshu/Desktop/School Year 12/Software engineering/Yr12 Assesment 3 Joshua.Muunoja SEN/JMEodel.keras")

def evaluateUsingLabeledData():
    model = tf.keras.models.load_model("C:/Users/joshu/Desktop/School Year 12/Software engineering/Yr12 Assesment 3 Joshua.Muunoja SEN/JMEodel.keras")
    image_number = 4
    while os.path.isfile(f"C:/Users/joshu\Desktop/School Year 12/Software engineering/Yr12 Assesment 3 Joshua.Muunoja SEN/28by28_Drawn/digit{image_number}.png"):
        try:
            img = cv2.imread(f"C:/Users/joshu\Desktop/School Year 12/Software engineering/Yr12 Assesment 3 Joshua.Muunoja SEN/28by28_Drawn/digit{image_number}.png")[:,:,0]
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

'''
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
'''
if (mode == "T"):
    trainNeuralNetwork();
elif (mode == "E"):
    evaluateUsingLabeledData();
elif (mode == "P"):
    predictUsingModel();




    
    #loss, accuracy = model.evaluate(x_test, y_test)

    #print(loss);
    #print(accuracy);