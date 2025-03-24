import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Load training data with domain labels
from sklearn.metrics import accuracy_score

import load_features

train_features, domain_labels = load_features.load_data_with_domain_label()

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

# Perform t-SNE to visualize the clusters
tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
train_features_tsne = tsne.fit_transform(train_features)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 8))
plt.scatter(train_features_tsne[:, 0], train_features_tsne[:, 1], c=new_labels)
plt.title(f"t-SNE Visualization of Clustered Data (Optimal k: {optimal_k}, Accuracy: {max_accuracy:.2f})")
plt.show()