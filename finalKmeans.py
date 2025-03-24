import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Load training data with domain labels
from sklearn.metrics import accuracy_score

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
train_features, domain_labels = load_data_with_domain_label()

# Find the optimal number of clusters using accuracy score
max_accuracy = 0
optimal_k = 0
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(train_features)

    # Convert cluster labels to discrete labels using majority vote
    unique_labels = np.unique(cluster_labels)
    new_labels = np.zeros_like(cluster_labels)
    for label in unique_labels:
        mask = cluster_labels == label
        mode = np.argmax(np.bincount(domain_labels[mask]))
        new_labels[mask] = mode

    # Calculate accuracy score
    accuracy = accuracy_score(domain_labels, new_labels)

    if accuracy > max_accuracy:
        max_accuracy = accuracy
        optimal_k = k

print(f"Optimal number of clusters: {optimal_k} (Accuracy: {max_accuracy:.2f})")

# Use K-Means to cluster the images based on domain labels using the optimal k
kmeans = KMeans(n_clusters=optimal_k)
cluster_labels = kmeans.fit_predict(train_features)

# Convert cluster labels to discrete labels using majority vote
unique_labels = np.unique(cluster_labels)
new_labels = np.zeros_like(cluster_labels)
for label in unique_labels:
    mask = cluster_labels == label
    mode = np.argmax(np.bincount(domain_labels[mask]))
    new_labels[mask] = mode

#accuracy
accuracy = accuracy_score(domain_labels, new_labels)
print('domain_labels')
print(domain_labels)
print('new_labels')
print(new_labels)


# Perform t-SNE to visualize the clusters
tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
train_features_tsne = tsne.fit_transform(train_features)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 8))
plt.scatter(train_features_tsne[:, 0], train_features_tsne[:, 1], c=new_labels)
plt.title(f"t-SNE Visualization of Clustered Data (Optimal k: {optimal_k}, Accuracy: {max_accuracy:.2f})")
plt.show()
