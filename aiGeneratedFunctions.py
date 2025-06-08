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
            #image = np.invert(np.array([image]))
            image = np.invert(image);
            np.array(image).reshape(28,28,1);
            
            #image = cv2.resize(img, (28, 28))  # Resize to **24×24 pixels**
            #image = image.astype(np.float32) / 255.0  # Normalize pixel values (0 to 1)

            label = label.replace('.png', '')
            label = label.replace('.jpeg', '')
            label = label.replace('.jpg', "")

            ascii_dict = {
            '0': ord('0'), '1': ord('1'), '2': ord('2'), '3': ord('3'), '4': ord('4'),
            '5': ord('5'), '6': ord('6'), '7': ord('7'), '8': ord('8'), '9': ord('9'),
            'A': ord('a'), 'B': ord('b'), 'C': ord('c'), 'D': ord('d'), 'E': ord('e'),
            'F': ord('f'), 'G': ord('g'), 'H': ord('h'), 'I': ord('i'), 'J': ord('j'),
            'K': ord('k'), 'L': ord('l'), 'M': ord('m'), 'N': ord('n'), 'O': ord('o'),
            'P': ord('p'), 'Q': ord('q'), 'R': ord('r'), 'S': ord('s'), 'T': ord('t'),
            'U': ord('u'), 'V': ord('v'), 'W': ord('w'), 'X': ord('x'), 'Y': ord('y'),
            'Z': ord('z'), 'ampersand': ord('&'), 'asterisk': ord('*'), 'at': ord('@'),
            'dollar': ord('$'), 'exclmark': ord('!'), 'greaterthan': ord('>'),
            'hash': ord('#'), 'lessthan': ord('<'), 'minus': ord('-'),
            'percent': ord('%'), 'plus': ord('+'), 'quesmark': ord('?')
            }

            char_to_id = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18,
            'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27,
            'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35,
            'ampersand': 36, 'asterisk': 37, 'at': 38, 'dollar': 39, 'exclmark': 40,
            'greaterthan': 41, 'hash': 42, 'lessthan': 43, 'minus': 44, 'percent': 45,
            'plus': 46, 'quesmark': 47
            }

            label = char_to_id[label];

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


