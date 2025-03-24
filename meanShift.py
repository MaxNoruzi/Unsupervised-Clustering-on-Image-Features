from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, accuracy_score

from typing import Tuple


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

# load data with domain label
train_features, domain_labels = load_data_with_domain_label()

# extract features and normalize them
features = train_features

# estimate bandwidth for Mean Shift algorithm
# bandwidth = estimate_bandwidth(features, quantile=0.2)

# cluster using Mean Shift
ms = MeanShift(bandwidth=10)
cluster_labels = ms.fit_predict(features)

# Convert cluster labels to discrete labels using majority vote
unique_labels = np.unique(cluster_labels)
new_labels = np.zeros_like(cluster_labels)
for label in unique_labels:
    mask = cluster_labels == label
    mode = np.argmax(np.bincount(domain_labels[mask]))
    new_labels[mask] = mode

# Calculate accuracy score
accuracy = accuracy_score(domain_labels, new_labels)

# get number of clusters
n_clusters = len(set(cluster_labels))

# assign domain label to each image based on cluster label
image_labels = []
for i in range(len(cluster_labels)):
    cluster_label = cluster_labels[i]
    domain_label = domain_labels[i]
    image_labels.append((cluster_label, domain_label))

# print number of images in each cluster
cluster_sizes = [0] * n_clusters
for i in range(len(cluster_labels)):
    cluster_sizes[cluster_labels[i]] += 1
for i in range(n_clusters):
    print(f'Number of images in cluster {i}: {cluster_sizes[i]}')

print(f'Silhouette score: {accuracy:.3f}')


# Plot clustering results
# Perform t-SNE to visualize the clusters
tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
train_features_tsne = tsne.fit_transform(train_features)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 8))
plt.scatter(train_features_tsne[:, 0], train_features_tsne[:, 1], c=new_labels)
plt.title(f"t-SNE Visualization of Clustered Data (Optimal k: {n_clusters}, Accuracy: {accuracy:.2f})")
plt.show()