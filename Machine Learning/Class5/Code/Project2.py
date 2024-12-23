import pandas as pd
import numpy as np

# Load the dataset
file_path = 'data/clustering2.csv'
adj_matrix = pd.read_csv(file_path, header=None).values

# Step 1: DBSCAN implementation with ε=1
def dbscan_custom(adj_matrix, min_pts_values):
    n = adj_matrix.shape[0]
    results = {}

    for min_pts in min_pts_values:
        visited = np.zeros(n, dtype=bool)  # To mark visited nodes
        cluster_labels = -np.ones(n, dtype=int)  # To assign clusters
        cluster_id = 0  # Cluster ID starts from 0
        core_objects = []

        # Step 2: Identify core objects
        for i in range(n):
            neighbors = np.where(adj_matrix[i] == 1)[0]
            if len(neighbors) >= min_pts:
                core_objects.append(i)

        # Step 3: Process each core object
        for core in core_objects:
            if visited[core]:
                continue
            visited[core] = True
            cluster_labels[core] = cluster_id

            # Initialize the neighbor set
            neighbors = set(np.where(adj_matrix[core] == 1)[0])
            neighbors_to_visit = neighbors.copy()

            while neighbors_to_visit:
                neighbor = neighbors_to_visit.pop()
                if not visited[neighbor]:
                    visited[neighbor] = True
                    cluster_labels[neighbor] = cluster_id
                    # Check if the neighbor is also a core object
                    neighbor_neighbors = np.where(adj_matrix[neighbor] == 1)[0]
                    if len(neighbor_neighbors) >= min_pts:
                        neighbors_to_visit.update(neighbor_neighbors)

            cluster_id += 1

        # Step 4: Compute clustering coefficients
        cluster_coefficients = {}
        for cluster in range(cluster_id):
            cluster_nodes = np.where(cluster_labels == cluster)[0]
            if len(cluster_nodes) > 1:
                subgraph = adj_matrix[np.ix_(cluster_nodes, cluster_nodes)]
                n_edges = np.sum(subgraph) / 2  # Count undirected edges
                n_nodes = len(cluster_nodes)
                cluster_coefficient = (2 * n_edges) / (n_nodes * (n_nodes - 1))
            else:
                cluster_coefficient = 0
            cluster_coefficients[cluster] = cluster_coefficient

        results[min_pts] = {
            "num_clusters": cluster_id,
            "cluster_coefficients": cluster_coefficients
        }

    return results

# Run the custom DBSCAN algorithm
min_pts_values = [10, 15, 20]
results = dbscan_custom(adj_matrix, min_pts_values)

# Print results
for min_pts, result in results.items():
    print(f"MinPts = {min_pts}")
    print(f"Number of clusters: {result['num_clusters']}")
    print("Cluster coefficients:")
    for cluster, coefficient in result["cluster_coefficients"].items():
        print(f"  Cluster {cluster}: {coefficient:.4f}")