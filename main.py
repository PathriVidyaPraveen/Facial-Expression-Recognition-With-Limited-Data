import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
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


# SVD for unlabelled data exploration

print("Performing SVD!!")

U,S,Vt = np.linalg.svd(X_unlabelled,full_matrices=True)
print("SVD completed!")
print("Shapes:")
print("U:", U.shape)     
print("S:", S.shape)     
print("Vt:", Vt.shape)   

# Save the matrices obtained in SVD for reuse
np.save("svd_U.npy", U)
np.save("svd_S.npy", S)
np.save("svd_Vt.npy", Vt)

# Part 2 : Feature Engineering with PCA 

# 1. Principal component analysis ( PCA )

# Range of PCA components to test
components_range = [20, 40, 60, 80, 100, 120, 150, 200]

# Store average cross-validation scores
mean_scores = []

for n_components in components_range:
    print(f"Testing PCA with {n_components} components...")

    # Step 1: Apply PCA
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    X_pca = pca.fit_transform(X_labelled)

    # Step 2: Train and evaluate SVM using cross-validation
    svm = SVC(kernel='rbf', C=1, gamma='scale')  # Keep SVM config fixed here
    scores = cross_val_score(svm, X_pca, y_labelled, cv=5, scoring='accuracy')
    mean_accuracy = scores.mean()
    mean_scores.append(mean_accuracy)

    print(f"Accuracy with {n_components} components: {mean_accuracy:.4f}")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(components_range, mean_scores, marker='o')
plt.xlabel("Number of PCA Components")
plt.ylabel("Cross-Validated Accuracy")
plt.title("PCA Components vs SVM Performance")
plt.grid(True)
plt.show()

# Best performing configuration
best_n = components_range[np.argmax(mean_scores)]
best_score = max(mean_scores)
print(f"\nBest number of components: {best_n} with accuracy {best_score:.4f}")



