import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  

# load the data
X_labelled = np.load("X_labelled.npy")
y_labelled = np.load("Y_labelled.npy")
X_unlabelled = np.load("X_unlabelled.npy")

# combine labelled and unlabelled data to learn a general face space
X_combined = np.vstack((X_labelled, X_unlabelled))

# perform PCA with 3 components for 3D visualization
pca = PCA(n_components=3, svd_solver='randomized', whiten=True)
pca.fit(X_combined)

# project only labelled data for visualization
X_labelled_pca_3d = pca.transform(X_labelled)

# get unique emotion labels for coloring
unique_emotions = np.unique(y_labelled)
palette = sns.color_palette("tab10", len(unique_emotions))
emotion_to_color = {emotion: palette[i] for i, emotion in enumerate(unique_emotions)}
colors = [emotion_to_color[label] for label in y_labelled]

# plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for emotion in unique_emotions:
    idx = (y_labelled == emotion)
    ax.scatter(
        X_labelled_pca_3d[idx, 0],
        X_labelled_pca_3d[idx, 1],
        X_labelled_pca_3d[idx, 2],
        label=emotion,
        alpha=0.7,
        s=40
    )

ax.set_title("3D PCA Projection of Labelled Faces (Trained on All Data)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.legend()
plt.tight_layout()
plt.savefig("pca_3d_projection.png", dpi=300)
plt.show()
