import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
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

# Emotions are defined as determined index below:

# 0 : Anger (45 samples)
# 1 : Disgust (59 samples)
# 2 : Fear (25 samples)
# 3 : Happiness (69 samples)
# 4 : Sadness (28 samples)
# 5 : Surprise (83 samples)
# 6 : Neutral (593 samples)
# 7 : Contempt (18 samples)

emotion_map = {
    0 : "Anger",
    1 : "Disgust",
    2 : "Fear",
    3 : "Happiness",
    4 : "Sadness",
    5 : "Surprise",
    6 : "Neutral",
    7 : "Contempt"
}


labelled = pd.read_csv("labelled_dataset_ck+.csv")
X_labelled = []
y_labelled = []

for index,row in labelled.iterrows():
    pixel_string = row["pixels"].strip()
    emotion = int(row["emotion"])
    emotion = emotion_map[emotion]

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

U,S,Vt = np.linalg.svd(X_unlabelled,full_matrices=False)
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



# 2 . Leveraging unlabelled data for PCA 

# train PCA on unlabelled data

pca = PCA(n_components=best_n , svd_solver='randomized' , whiten=True)
pca.fit(X_unlabelled) # learn on general face space

# transform labelled data using unlabelled pca

X_labelled_pca = pca.transform(X_labelled)
# save for reuse
np.save("X_labelled_pca_from_unlabelled_basis.npy", X_labelled_pca)

# Part 3 : SVM Classification and evaluation 

# 1 . Training the SVM 

# initialize SVC ie SVM classifier

svm = SVC()

# define parameter grid for hyperparameter tuning
parameter_grid = {
    'kernel': ['linear' , 'rbf' , 'poly'],
    'C':[0.1,1,10],
    'gamma': ['scale','auto'],
    'degree':[2,3] 
}

# grid search with 5 fold cross validation

grid_search = GridSearchCV(svm , parameter_grid , cv=5 , scoring='accuracy', verbose=2 , n_jobs=-1)

# fit the model
grid_search.fit(X_labelled_pca, y_labelled)

# Print the best model and score
print("Best Parameters:")
print(grid_search.best_params_)

print("Best Cross-Validation Accuracy:")
print(grid_search.best_score_)

# evaluate on full data
y_pred = grid_search.predict(X_labelled_pca)

print("Classification Report on Training Data:")
print(classification_report(y_labelled, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_labelled, y_pred))

# 2. Evaluation

X = X_labelled_pca
y = y_labelled

# split into training and testing sets (70/30) , stratified

X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)

# run the grid search on training data only
svm_eval = SVC()

grid_search_eval = GridSearchCV(svm_eval , parameter_grid , cv=5 , scoring='accuracy' , verbose=2 , n_jobs=-1)

grid_search_eval.fit(X_train,y_train)

print("Using best params on test set evaluation:")
print(grid_search_eval.best_params_)


best_svm = SVC(**grid_search_eval.best_params_)

best_svm.fit(X_train, y_train)

# predict on test set
y_pred = best_svm.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# precision , recall , f1 score
print("Classification Report:")
print(classification_report(y_test, y_pred))

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# visualization if 2d or 3d after PCA

if X.shape[1] == 2:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', edgecolor='k')
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA Projection (2D)")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True)
    plt.show()

elif X.shape[1] == 3:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='tab10', edgecolor='k')
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    plt.title("PCA Projection (3D)")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()

