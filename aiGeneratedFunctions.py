import os
import cv2
import numpy as np

def load_training_data(directory="trainingData"):
    """
    Loads images from the specified directory, extracts their labels from filenames,
    converts them into NumPy arrays, and returns image data with corresponding labels.
    
    Args:
        directory (str): Path to the directory containing labeled images.
    
    Returns:
        tuple: (numpy array of images, numpy array of labels)
    """
    images = []
    labels = []
    
    # Ensure the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")

    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Check for valid image formats
            filepath = os.path.join(directory, filename)
            
            #Split string that is file path of img
            parts = filename.split("_")

            # Extract character label
            label = parts[len(parts)-1]  
            
            # Load image using OpenCV and convert to a NumPy array
            image = cv2.imread(filepath)[:,:,0]
            image = np.invert(np.array([image]))
            #image = cv2.resize(img, (28, 28))  # Resize to **24×24 pixels**
            #image = image.astype(np.float32) / 255.0  # Normalize pixel values (0 to 1)

            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)


def load_nonlabeled_data(directory="nonLabeledInput"):
    """
    Loads images from the specified directory, extracts their labels from filenames,
    converts them into NumPy arrays, and returns image data with corresponding labels.
    
    Args:
        directory (str): Path to the directory containing labeled images.
    
    Returns:
        tuple: (numpy array of images, numpy array of labels)
    """
    images = []
    labels = []
    
    # Ensure the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")

    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Check for valid image formats
            filepath = os.path.join(directory, filename)
            
            # Load image using OpenCV and convert to a NumPy array

            image = cv2.imread(filepath)[:,:,0]
            image = np.invert(np.array([image]))

            #image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Load in grayscale for CNN input
            #image = cv2.resize(image, (28, 28))  # Resize to **24×24 pixels**
            #image = image.astype(np.float32) / 255.0  # Normalize pixel values (0 to 1)

            images.append(image)

    return np.array(images)


def print_image_array(image_array):
    """
    Prints the 2D array of pixel values from a given image numpy array.
    Handles both single images and batches (with shape [1, H, W] or [H, W]).
    """
    # If the image is wrapped in an extra dimension (e.g., [1, H, W]), squeeze it
    if image_array.ndim == 3 and image_array.shape[0] == 1:
        image_array = image_array[0]
    
    for row in image_array:
        print(' '.join(f'{val:3}' for val in row))


