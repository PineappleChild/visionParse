import torch

from pathlib import Path

from data.loader import dataLoad

from utils.tensor_utils import denormalize_imagenet

from utils.visualization import visualize_pca_of_images, visualize_clusters_of_image

from scoring.scoring_with_clip import create_CLIP_score_arr
from scoring.scoring_with_dinov2 import score_clusters_with_dino_cls

from clustering.kmeans import cluster_features_kmeans

from models.dinov2 import get_dinov2_model, extract_dinov2_patch_features
from models.clip import get_clip_model, build_clip_prompts


from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from pycocotools import mask as mask_utils

# clustering is the biggest bottneck in getting good regions and predictions for what is not part of the target prompts
#smoother and less chunky would be good, also sometimes it bleeds in and causes it to change the scores for the background and 
#mixed in target area resulting in a poor score

def main_preprocess_loop(
                        data_loader,
                        device
                        ):

    dinov2_model = get_dinov2_model(model_name="dinov2_vitl14_reg")
    
    clip_model, clip_preprocess = get_clip_model(model_name="ViT-B/32", device=device)
    target_prompts, background_prompts = build_clip_prompts(clip_model=clip_model, device=device)

    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"Starting batch {batch_idx}")
        # images = torch.clamp((images.cpu() + 1) / 2.0, 0, 1)
        # image = image.permute(1, 2, 0).numpy()
        batch_of_images_for_pca = []
        batch_of_images_for_clustering = []
        clusters_scores_cls = []
        clusters_scores_CLIP = []
        for idx, image in enumerate(images):
            print(f"--------------[IMAGE {idx + 1}]--------------")
            feature_patch_info_dict = extract_dinov2_patch_features(image, dinov2_model)
            batch_of_images_for_pca.append((feature_patch_info_dict["features"], feature_patch_info_dict["orig_image"]))

            clusters_dict = cluster_features_kmeans(feature_patch_info_dict)
            batch_of_images_for_clustering.append((clusters_dict, feature_patch_info_dict))
            #using dino feature vectors
            scores = score_clusters_with_dino_cls(feature_patch_info_dict, clusters_dict)
            clusters_scores_cls.append(scores)

            #using clip and some prompts
            reference_image = denormalize_imagenet(image)
            clusters_scores_CLIP.append(create_CLIP_score_arr(clusters_dict["clusters"], reference_image, 
                                                            clip_model, clip_preprocess,
                                                            device, target_prompts, background_prompts,))

            
        visualize_clusters_of_image(batch_of_images_for_clustering, clusters_scores_cls)

        visualize_clusters_of_image(batch_of_images_for_clustering, clusters_scores_CLIP)

        visualize_pca_of_images(batch_of_images_for_pca)

        #to cut batch off early for testing
        print(f"Donw with batch {batch_idx}")
        return None
    
    

path_to_train = Path(r"E:\featureLearningJourney\LatentSpaceFruitsAndVeggies\validation")

image_size = 980
#min 2 for now
batch_size = 2

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    else:
        print("CUDA not available")
    
    dataloader_train, class_names_train = dataLoad(  
        file_path=path_to_train,
        resize_dim=image_size,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    main_preprocess_loop(data_loader=dataloader_train, device=device)