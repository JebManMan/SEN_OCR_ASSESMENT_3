import sys
import os
import math
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
    QMessageBox,
    QFileDialog,
)
from PyQt6.QtCore import (
    Qt,
    QPropertyAnimation,
    QEasingCurve,
    QObject,
    QThread,
    pyqtSignal,
)
from PyQt6.QtGui import QFont

# Import your preexisting functions (assuming they can now accept a file path)
from aiGeneratedFunctions import load_training_data, load_nonlabeled_data, print_image_array

# Global mapping dictionary (unchanged)
idToName = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i',
    19: 'j', 20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r',
    28: 's', 29: 't', 30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z',
    36: 'ampersand', 37: 'asterisk', 38: 'at', 39: 'dollar', 40: 'exclmark',
    41: 'greaterthan', 42: 'hash', 43: 'lessthan', 44: 'minus', 45: 'percent',
    46: 'plus', 47: 'quesmark'
}

#Gnerated on the 9/06/2025 using copiolet
#Modified and reviewed
def display_predictions(model, imagesToPredict, idToName):
    num_images = len(imagesToPredict)
    if num_images == 0:
        print("No images provided.")
        return

    # Determine grid size based on the number of images
    nrows = int(math.ceil(math.sqrt(num_images)))
    ncols = int(math.ceil(num_images / nrows))
    
    # Create a figure with the appropriate number of subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    
    # If only one image, ensure axes is always iterable
    if num_images == 1:
        axes = [axes]
    else:
        # Flatten in case we have a 2D array of axes
        axes = axes.flatten()

    # Loop over images and make predictions
    for i, image in enumerate(imagesToPredict):
        ax = axes[i]
        try:
            # Get prediction for the image
            prediction = model.predict(image)
            predicted_class = int(np.argmax(prediction))
            predicted_label = idToName[predicted_class]
            
            # Display image and predicted label
            ax.imshow(image[0], cmap='binary')
            ax.set_title(predicted_label)
            ax.axis('off')
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            ax.text(0.5, 0.5, "Error", ha="center", va="center", transform=ax.transAxes)
            ax.axis('off')
    
    # Hide any extra subplots in the grid
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
plt.show()

# Modified preexisting functions that now accept a file path parameter.
def trainNeuralNetwork(file_path="__Image Files__"):
    # Load training data from the chosen file path.
    x_data, y_data = load_training_data(file_path)
    print("Training using file path:", file_path)
    print_image_array(x_data[0])
    print(y_data)
    
    x_train = x_data
    y_train = y_data
    print_image_array(x_train[-1])
    
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(28, 28, 1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(70, activation='relu'))
    model.add(tf.keras.layers.Dense(48, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    model.save("JMEodel.keras")
    print("Model training complete and saved.")


def evaluateUsingLabeledData(file_path="28by28_Drawn"):
    # Load the model and evaluation data.
    model = tf.keras.models.load_model("JMEodel.keras")
    print("Evaluating using file path:", file_path)
    imagesToPredict = load_nonlabeled_data(file_path)
    display_predictions(model, imagesToPredict, idToName)


def predictUsingModel(file_path="default_predict_path"):
    # For demonstration we just print a message.
    print("Predicting using file path:", file_path)
    # Insert your prediction code here.
    # For example:
    # model = tf.keras.models.load_model("JMEodel.keras")
    # imagesToPredict = load_nonlabeled_data(file_path)
    # display_predictions(model, imagesToPredict, idToName)


# ----------------------
# Worker for background execution
# ----------------------
class Worker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


# ----------------------
# Main GUI Window
# ----------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Caligarde the Character Wizard")
        self.setGeometry(100, 100, 400, 300)

        # Central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Title label
        self.title_label = QLabel("Caligarde the Character Wizard", self)
        title_font = QFont("Arial", 18, QFont.Weight.Bold)
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        # Buttons
        self.predict_button = QPushButton("Predict", self)
        self.evaluate_button = QPushButton("Evaluate", self)
        self.train_button = QPushButton("Train", self)

        for btn in [self.predict_button, self.evaluate_button, self.train_button]:
            btn.setFixedHeight(40)
            btn.setFixedWidth(200)
            layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Connect buttons to methods
        self.predict_button.clicked.connect(self.on_predict)
        self.evaluate_button.clicked.connect(self.on_evaluate)
        self.train_button.clicked.connect(self.on_train)

        # Perform a slow "pop in" animation via changing window opacity.
        self.setWindowOpacity(0)
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(1000)  # Duration in milliseconds.
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.animation.start()

    def disable_ui(self):
        self.predict_button.setEnabled(False)
        self.evaluate_button.setEnabled(False)
        self.train_button.setEnabled(False)

    def enable_ui(self):
        self.predict_button.setEnabled(True)
        self.evaluate_button.setEnabled(True)
        self.train_button.setEnabled(True)

    def ask_file_path(self, default_path):
        # Ask user if they want to change the file path.
        reply = QMessageBox.question(
            self,
            "Change File Path",
            "Would you like to change the file path?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            # Launch a dialog to select a directory. Modify to getOpenFileName if appropriate.
            selected_path = QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
            if selected_path:
                return selected_path
            else:
                return default_path
        else:
            return default_path

    def run_function_in_thread(self, func, *args, **kwargs):
        self.disable_ui()
        self.thread = QThread()
        self.worker = Worker(func, *args, **kwargs)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.error.connect(self.handle_error)
        self.thread.finished.connect(self.enable_ui)
        self.thread.start()

    def handle_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)

    def on_train(self):
        # Ask and possibly update the file path for training.
        file_path = self.ask_file_path("__Image Files__")
        self.run_function_in_thread(trainNeuralNetwork, file_path)

    def on_evaluate(self):
        # Ask and possibly update the file path for evaluation.
        file_path = self.ask_file_path("28by28_Drawn")
        self.run_function_in_thread(evaluateUsingLabeledData, file_path)

    def on_predict(self):
        # Ask and possibly update the file path for prediction.
        file_path = self.ask_file_path("default_predict_path")
        self.run_function_in_thread(predictUsingModel, file_path)

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
