
from typing import Tuple

from sklearn.manifold import TSNE
# import load_features as lf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score


# Load data with domain labels

def load_data(train_data: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    loads train/test features with image labels.
    """
    if train_data:
        data = np.load(f'train_data.npz')
    else:
        data = np.load(f'test_data.npz')

    features = data['features']
    img_labels = data['img_labels']

    return features, img_labels


def load_data_with_domain_label() -> Tuple[np.ndarray, np.ndarray]:
    """
    loads portion of training features with domain label
    """
    data = np.load(f'train_data_w_label.npz')
    train_features = data['features']
    domain_labels = data['domain_labels']

    return train_features, domain_labels


# Train Data with image labels
train_features, image_labels = load_data(True)
print(train_features.shape)
print(image_labels.shape)
print(f'Number of training samples: {train_features.shape[0]}')

# Test Data with image labels
test_features, image_labels = load_data(False)
print(test_features.shape)
print(image_labels.shape)

# 5% of train data with domain label
train_features, image_labels = load_data_with_domain_label()
print(train_features.shape)
print(image_labels.shape)

train_features, domain_labels = load_data_with_domain_label()


# %%
# Define range of eps and min_samples values to try
eps_range = np.linspace(10, 50, 9)
min_samples_range = range(1, 6)

best_accuracy = 0
best_eps = 0
best_min_samples = 0

for eps in eps_range:
    for min_samples in min_samples_range:
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(train_features)

        # Convert cluster labels to discrete labels using majority vote
        unique_labels = np.unique(cluster_labels)
        new_labels = np.zeros_like(cluster_labels)
        for label in unique_labels:
            mask = cluster_labels == label
            mode = np.argmax(np.bincount(domain_labels[mask]))
            new_labels[mask] = mode

        # Calculate accuracy score
        accuracy = accuracy_score(domain_labels, new_labels)

        # Print information about the current iteration
        n_clusters = len(set(cluster_labels))
        print(f"eps: {eps:.3f}, min_samples: {min_samples}, accuracy: {accuracy:.3f}, n_clusters: {n_clusters}")

        # Update best parameters if accuracy has improved while keeping the number of clusters low
        if accuracy > best_accuracy and n_clusters <= 20:
            best_accuracy = accuracy
            best_eps = eps
            best_min_samples = min_samples


print(f"Best parameters: eps={best_eps:.3f}, min_samples={best_min_samples}, accuracy={best_accuracy:.3f}")



# %%
# Apply DBSCAN clustering with the best parameters
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
cluster_labels = dbscan.fit_predict(train_features)

# %%
# Convert cluster labels to discrete labels using majority vote
unique_labels = np.unique(cluster_labels)
new_labels = np.zeros_like(cluster_labels)
for label in unique_labels:
    mask = cluster_labels == label
    mode = np.argmax(np.bincount(domain_labels[mask]))
    new_labels[mask] = mode

# Print number of images in each cluster
n_clusters = len(set(cluster_labels))
cluster_sizes = [0] * n_clusters
for i in range(len(cluster_labels)):
    cluster_sizes[cluster_labels[i]] += 1
for i in range(n_clusters):
    print(f'Number of images in cluster {i}: {cluster_sizes[i]}')

# %%
# Plot clustering results
# Perform t-SNE to visualize the clusters
tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
train_features_tsne = tsne.fit_transform(train_features)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 8))
plt.scatter(train_features_tsne[:, 0], train_features_tsne[:, 1], c=new_labels)
plt.title(f"t-SNE Visualization of Clustered Data (Optimal k: {n_clusters}, Accuracy: {best_accuracy:.2f})")
plt.show()


# %%



