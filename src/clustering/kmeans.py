from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F

import numpy as np

from clustering.selection import estimate_optimal_num_clusters
from clustering.refinement import filter_small_clusters

def cluster_features_kmeans(
                    feature_dict: dict,
                    num_clusters: int = None
                    ) -> dict:
    
    features = feature_dict["features"]
    features_h = feature_dict["features_h"]
    features_w = feature_dict["features_w"]
    orig_image = feature_dict["orig_image"]
    orig_image_w, orig_image_h = orig_image.shape[1], orig_image.shape[2]

    normalized_features = normalize(features, norm="l2")

    n_pca = min(32, features.shape[0] - 1, features.shape[1])
    pca = PCA(n_components=n_pca, random_state=42)
    features_reduced = pca.fit_transform(normalized_features)

    normalized_features = normalize(features_reduced, norm="l2")

    grid_y = np.linspace(0, 1, features_h).repeat(features_w).reshape(-1, 1)
    grid_x = np.tile(np.linspace(0, 1, features_w), features_h).reshape(-1, 1)
    spatial_weight = 0.2   
    features_spatial = np.hstack([
        normalized_features,
        grid_y * spatial_weight,
        grid_x * spatial_weight
    ])

    if num_clusters is None:
        num_clusters = estimate_optimal_num_clusters(features, 2, 6)

    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_spatial)
    labels = filter_small_clusters(labels, min_fraction=0.03)

    unique_labels = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([remap[l] for l in labels])
    num_clusters = len(unique_labels)

    clusters_map = labels.reshape(features_h, features_w)

    clusters_map_tensor = torch.tensor(clusters_map).float().unsqueeze(0).unsqueeze(0)

    clusters_map_up = F.interpolate(clusters_map_tensor, size=(orig_image_h, orig_image_w), mode="nearest").squeeze().long().numpy()

    clusters = [[] for i in range(num_clusters)]
    for label in range(num_clusters):
        clusters[label].append((clusters_map_up == label).astype(int))

    cluster_dict = {
        "clusters_map": clusters_map,
        "clusters_map_up": clusters_map_up,
        "clusters": clusters,
        "centers": kmeans.cluster_centers_,
        "labels": labels,
        "num_clusters": num_clusters
    }

    return cluster_dict