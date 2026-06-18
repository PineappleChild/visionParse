
def score_clusters_with_dino_cls(
                                feature_dict: dict,
                                cluster_dict: dict
                                ) -> list:
    
    patch_features = feature_dict["features"]   
    cls_feature = feature_dict["cls_token"]  

    patch_sims = patch_features @ cls_feature   

    labels = cluster_dict["labels"]
    num_clusters = cluster_dict["num_clusters"]

    scores = []
    for cluster_id in range(num_clusters):
        mask = labels == cluster_id
        score = float(patch_sims[mask].mean()) if mask.sum() > 0 else 0.0
        scores.append((cluster_id, score))

    print(f"[CLS] cluster scores: { {i: f'{s:.4f}' for i, s in scores} }")
    return scores