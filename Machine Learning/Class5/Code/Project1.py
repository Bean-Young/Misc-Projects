import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Step 1: Load the dataset
file_path = 'data/clustering1.csv'
data = pd.read_csv(file_path).values

# Step 2: Min-Max Normalization
def min_max_normalize(data):
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

normalized_data = min_max_normalize(data)

# Step 3: K-means clustering
def kmeans_clustering(data, k, max_iter=100, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    # Randomly initialize cluster centers
    n_samples, n_features = data.shape
    centers = data[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iter):
        # Assign clusters based on the closest center
        distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        cluster_labels = np.argmin(distances, axis=1)

        # Recompute cluster centers
        new_centers = np.array([data[cluster_labels == i].mean(axis=0) for i in range(k)])
        
        # Stop if centers do not change
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return cluster_labels, centers

# Step 4: PCA for dimensionality reduction
def pca(data, n_components=2):
    # Center the data
    mean = data.mean(axis=0)
    centered_data = data - mean

    # Compute covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by descending eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]

    # Project data onto top components
    reduced_data = centered_data.dot(top_eigenvectors)
    return reduced_data

# Step 5: Visualization and save
def kmeans_clustering_and_save(data, k_values, max_iter=100, save_path="figure/"):
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(len(k_values), 2, figsize=(12, 12))
    for i, k in enumerate(k_values):
        for j in range(2):  # Two different random initializations
            cluster_labels, _ = kmeans_clustering(data, k, max_iter=max_iter, random_state=j)

            # Reduce data to 2D using PCA
            reduced_data = pca(data, n_components=2)

            # Plot the clustered data in 2D
            ax = axes[i, j]
            for cluster in np.unique(cluster_labels):
                cluster_points = reduced_data[cluster_labels == cluster]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', alpha=0.6)
            ax.set_title(f'k={k}, Init={j+1}')
            ax.legend()
    
    plt.tight_layout()
    file_name = os.path.join(save_path, "kmeans_results.png")
    plt.savefig(file_name)  # Save the figure
    plt.show()
    print(f"Figure saved at: {file_name}")

# Step 6: Apply the function with k=2, 3, 5
kmeans_clustering_and_save(normalized_data, k_values=[2, 3, 5])