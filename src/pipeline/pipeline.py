import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from models.dinov2 import get_dinov2_model, extract_dinov2_patch_features
from models.clip import get_clip_model, build_clip_prompts
from models.sam import get_sam

from clustering.kmeans import cluster_features_kmeans
from clustering.selection import get_target_clusters

from scoring.scoring_with_clip import create_CLIP_score_arr, classify_clusters
from scoring.scoring_with_dinov2 import score_clusters_with_dino_cls

from utils.tensor_utils import denormalize_imagenet
from utils.visualization import visualize_pca_of_images, visualize_clusters_of_image, visualize_segmentation_of_image

from segmentation.segment_sam import segmenting_with_sam

import gc
import numpy as np
import torch

class Pipeline:
    def __init__(self, data_loader, device, **kwargs):
        self.device = device
        self.data_loader = data_loader

        self.result = []

        self.dinov2_model = get_dinov2_model(model_name="dinov2_vitl14_reg")
        self.clip_model, self.clip_preprocess = get_clip_model(model_name="ViT-B/32", device=device)
        self.target_prompts, self.background_prompts = build_clip_prompts(clip_model=self.clip_model, device=device)
        self.sam = get_sam(model_name="vit_b", checkpoint_path=kwargs["sam_checkpoint_path"], device=device)

    def run(self):
        for batch_idx, (images, labels) in enumerate(self.data_loader):
            print(f"Starting batch {batch_idx}")

            batch_results  = {
                "batch_idx": batch_idx,
                "batch_of_images_for_pca": [],
                "batch_of_images_for_clustering": [],
                "clusters_scores_cls": [],
                "clusters_scores_CLIP": [],
                "batch_of_masks_for_segmentation": []
            }

            for idx, image in enumerate(images):
                
                print(f"--------------[IMAGE {idx + 1}]--------------")
                print(len(batch_results["batch_of_images_for_pca"]))
                denormalize_image = denormalize_imagenet(image)

                with torch.no_grad():
                    feature_patch_info_dict = extract_dinov2_patch_features(image, self.dinov2_model)
                    batch_results["batch_of_images_for_pca"].append((feature_patch_info_dict["features"], feature_patch_info_dict["orig_image"]))

                    clusters_dict = cluster_features_kmeans(feature_patch_info_dict)
                    batch_results["batch_of_images_for_clustering"].append((clusters_dict, feature_patch_info_dict))

                    #using dino feature vectors
                    scores_dino_cls = score_clusters_with_dino_cls(feature_patch_info_dict, clusters_dict)
                    batch_results["clusters_scores_cls"].append(scores_dino_cls)

                    #using clip and some prompts
                    #change the scoring with clip to just use a loop rather than a helper to go over all the individual images
                    scores_clip = create_CLIP_score_arr(clusters_dict["clusters"], denormalize_image, 
                                                        self.clip_model, self.clip_preprocess,
                                                        self.device, self.target_prompts, self.background_prompts)
                                                                                
                    batch_results["clusters_scores_CLIP"].append(scores_clip)

                    classified_clusters = classify_clusters(clip_scores=scores_clip)

                    classified_clusters = get_target_clusters(clusters_dict=clusters_dict, classified_clusters=classified_clusters)

                    print(classified_clusters)
                    print(classified_clusters["target_clusters_idx"])

                    final_mask = segmenting_with_sam(denormalize_image, classified_clusters, self.sam)

                    batch_results["batch_of_masks_for_segmentation"].append((final_mask, feature_patch_info_dict["orig_image"]))

            self.result.append(batch_results["batch_of_masks_for_segmentation"])

            # visualize_clusters_of_image(batch_results["batch_of_images_for_clustering"], batch_results["clusters_scores_cls"])

            visualize_clusters_of_image(batch_results["batch_of_images_for_clustering"], batch_results["clusters_scores_CLIP"])

            visualize_segmentation_of_image(batch_results["batch_of_masks_for_segmentation"])

            # visualize_pca_of_images(batch_results["batch_of_images_for_pca"])

            del batch_results
            gc.collect()

            #to cut batch off early for testing
            # print(f"Donw with batch {batch_idx}")
            # return None