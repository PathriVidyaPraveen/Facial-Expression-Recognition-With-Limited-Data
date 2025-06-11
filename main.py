import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.datasets import fetch_lfw_people

# Part 1 : Data preprocessing and initial insights

# Image preprocessing

def preprocess_image(image, size=(64, 64)):
    # Validate input
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image: None or not a NumPy array")

    # If grayscale (2D), convert to 3 channels for consistent resizing
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Ensure image has 3 channels now
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Invalid image shape for color: {image.shape}")

    # Resize
    try:
        resized_image = cv2.resize(image, size)
    except Exception as e:
        raise ValueError(f"cv2.resize() failed: {e}")

    # Convert to grayscale (now it works for both sources)
    try:
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        raise ValueError(f"cv2.cvtColor() failed: {e}")

    # Normalize and flatten
    flattened_image = grayscale_image.flatten() / 255.0
    return flattened_image


# Labelled dataset 

emotion_map = {

}

labelled = pd.read_csv("labelled_dataset_ck+.csv")
X_labelled = []
y_labelled = []

for index,row in labelled.iterrows():
    pixel_string = row["pixels"].strip()
    emotion = int(row["emotion"])

    pixel_list = [int(a) for a in pixel_string.split()]
    pixels = np.array(pixel_list , dtype=np.uint8)

    image = pixels.reshape((48,48))

    # preprocess the image

    processed_image = preprocess_image(image)
    X_labelled.append(processed_image)
    y_labelled.append(emotion)

X_labelled = np.array(X_labelled)
y_labelled = np.array(y_labelled)

np.save("X_labelled.npy",X_labelled)
np.save("Y_labelled.npy",y_labelled)



# Take unlabelled LFW dataset and fetch images from it

lfw_dataset = fetch_lfw_people(color=True,resize=1.0) # use original images
unlabelled_images = lfw_dataset.images # shape: (n_samples, height, width, 3)
# print(unlabelled_images.shape)

# Now apply image preprocessing for unlabelled images
X_unlabelled = []

for idx, image in enumerate(unlabelled_images):
    try:
        # Check for None or non-array
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image format")

        # Convert float32 images (0–1) to uint8 (0–255)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Some images might have a broken shape
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        processed_image = preprocess_image(image)
        X_unlabelled.append(processed_image)

    except Exception as e:
        print(f"Error processing LFW image {idx}: {e}")
X_unlabelled = np.array(X_unlabelled)
np.save("X_unlabelled.npy",X_unlabelled)

