import torch

from PIL import Image

import numpy as np

def score_cluster_with_clip(
                        image,
                        mask,
                        clip_model,
                        clip_preprocess,
                        produce_features,
                        background_features,
                        device
                        ):
    
    image = np.clip(image.astype(np.float32), 0.0, 1.0)
    
    mask = np.array(mask).squeeze(0).astype(bool)

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    if (x_max - x_min) < 10 or (y_max - y_min) < 10:
        return 0.0

    pad = 10
    h, w = image.shape[:2]
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w - 1, x_max + pad)
    y_max = min(h - 1, y_max + pad)

    crop = image[y_min:y_max+1, x_min:x_max+1].copy()
    seg_crop = mask[y_min:y_max+1, x_min:x_max+1]
    
    crop[~seg_crop] = 1.0

    pil_crop = Image.fromarray((crop * 255).astype(np.uint8))

    image_input = clip_preprocess(pil_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        produce_sim = (image_features @ produce_features.T).mean(dim=-1)
        background_sim = (image_features @ background_features.T).mean(dim=-1)

        logit_scale = clip_model.logit_scale.exp()

        logits = logit_scale * torch.stack([produce_sim, background_sim], dim=-1)
        probs  = logits.softmax(dim=-1)

        produce_score = probs[0, 0].item()

        print(f"  [CLIP] (produce_sim={produce_sim.item():.4f} bg_sim={background_sim.item():.4f}) scale={logit_scale.item():.1f} score={produce_score:.4f}")

    return produce_score

def create_CLIP_score_arr(
                            clusters,
                            reference_image,
                            clip_model,
                            clip_preprocess,
                            device,
                            target_prompts: list,
                            background_prompts:list, 
                            ) -> list:
    cluster_scores_CLIP = []
    for idx, cluster in enumerate(clusters):
        score_for_mask_cluster = score_cluster_with_clip(
                            reference_image,
                            cluster,
                            clip_model,
                            clip_preprocess,
                            target_prompts,
                            background_prompts,
                            device
                            )
        
        cluster_scores_CLIP.append((idx, score_for_mask_cluster))
    return cluster_scores_CLIP