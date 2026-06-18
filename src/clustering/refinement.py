import numpy as np
import scipy

def filter_small_clusters(
                            labels: np.ndarray,
                            min_fraction: float = 0.02
                            ) -> np.ndarray:
    total = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    small = set(unique[counts < total * min_fraction])
    
    if not small:
        return labels
    
    filtered = labels.copy()
    large_labels = unique[counts >= total * min_fraction]
    
    for i, label in enumerate(filtered):
        if label in small:
            filtered[i] = -1  
    
    mask = filtered == -1
    if mask.any():
        indices = scipy.ndimage.distance_transform_edt(
            mask, return_distances=False, return_indices=True
        )
        filtered = filtered[tuple(indices)] if filtered.ndim > 1 else filtered
        # simpler 1D version:
        valid_idx = np.where(~mask)[0]
        for i in np.where(mask)[0]:
            nearest = valid_idx[np.argmin(np.abs(valid_idx - i))]
            filtered[i] = labels[nearest]
    
    return filtered