import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# load the data
X_labelled = np.load("X_labelled.npy")
y_labelled = np.load("Y_labelled.npy")
X_unlabelled = np.load("X_unlabelled.npy")

# combine both datasets to learn general face space
X_combined = np.vstack((X_labelled, X_unlabelled))

# perform PCA to result in 2 components (2d visualization)
pca = PCA(n_components=2, svd_solver='randomized', whiten=True)
pca.fit(X_combined)

# project only labelled data for visualization
X_labelled_pca_2d = pca.transform(X_labelled)

# unique emotion labels for consistent coloring
unique_emotions = np.unique(y_labelled)

# create color palette
palette = sns.color_palette("tab10", len(unique_emotions))
emotion_to_color = {emotion: palette[i] for i, emotion in enumerate(unique_emotions)}
colors = [emotion_to_color[label] for label in y_labelled]

# plotting
plt.figure(figsize=(10, 6))
for emotion in unique_emotions:
    idx = (y_labelled == emotion)
    plt.scatter(
        X_labelled_pca_2d[idx, 0],
        X_labelled_pca_2d[idx, 1],
        label=emotion,
        alpha=0.7,
        s=40
    )

plt.title("2D PCA Projection of Labelled Faces (Trained on All Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_2d_projection.png", dpi=300)
plt.show()
