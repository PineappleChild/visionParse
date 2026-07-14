import numpy as np

from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

def estimate_optimal_num_clusters(
                        features: np.ndarray,
                        min_clusters:int,
                        max_clusters:int,
                        ) -> int:
    
    normalized_features = normalize(features)

    num_samples = min(1000, len(normalized_features))
    idx = np.random.choice(len(normalized_features), num_samples, replace=False)
    sample = normalized_features[idx]

    best_num_clusters = None
    best_score = -1

    for i in range(min_clusters, max_clusters + 1):
        kmeans = MiniBatchKMeans(n_clusters=i, random_state=42, n_init=5, batch_size=512)
        labels = kmeans.fit_predict(sample)

        if(len(np.unique(labels)) < 2):
            continue

        score = silhouette_score(sample, labels)

        if(score > best_score):
            best_score = score
            best_num_clusters = i

    print(f"cluster_range={min_clusters},{max_clusters}, selected_num_clusters={best_num_clusters}")

    return best_num_clusters

def get_target_clusters(
                        clusters_dict: list,
                        classified_clusters: list
                        ) -> list:
    
    target_clusters = []
    background_clusters = []

    for idx in range(clusters_dict["num_clusters"]):
        if classified_clusters[idx][1][0] != "target":
            background_clusters.append(idx)
        else:
            target_clusters.append(idx)

    classified_cluster_dict = {
        "clusters": clusters_dict["clusters"],
        "clusters_map_up": clusters_dict["clusters_map"],
        "cluster_centers" : clusters_dict["centers"],
        "target_clusters_idx": target_clusters,
        "background_clusters_idx": background_clusters
    }

    return classified_cluster_dict