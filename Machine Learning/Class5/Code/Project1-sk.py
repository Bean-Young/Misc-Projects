import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

# Step 1: Load the dataset
file_path = 'data/clustering1.csv'
data = pd.read_csv(file_path)

# Step 2: Min-Max Normalization
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# Step 3: Define a function to perform K-means clustering and visualize
def kmeans_clustering_and_save(data, k_values, max_iter=100, save_path="figure/"):
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist
    
    fig, axes = plt.subplots(len(k_values), 2, figsize=(12, 12))
    for i, k in enumerate(k_values):
        for j in range(2):  # Two different random initializations
            kmeans = KMeans(n_clusters=k, max_iter=max_iter, random_state=j)
            cluster_labels = kmeans.fit_predict(data)

            # Step 4: PCA for dimensionality reduction to 2D
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(data)

            # Plot the clustered data in 2D
            ax = axes[i, j]
            for cluster in np.unique(cluster_labels):
                cluster_points = reduced_data[cluster_labels == cluster]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', alpha=0.6)
            ax.set_title(f'k={k}, Init={j+1}')
            ax.legend()
    
    plt.tight_layout()
    file_name = os.path.join(save_path, "kmeans_results_sklearn.png")
    plt.savefig(file_name)  # Save the figure
    plt.show()
    print(f"Figure saved at: {file_name}")

# Step 5: Apply the function with k=2, 3, 5
kmeans_clustering_and_save(normalized_data, k_values=[2, 3, 5])