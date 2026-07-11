import torch
from torch import Tensor

def get_dinov2_model(
                    model_name: str
                    ):
    dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
    dinov2.eval().cuda()

    return dinov2

def extract_dinov2_patch_features(
                            image: Tensor,
                            dinov2_model
                            ) -> list:
    
    # print(image.shape, type(image))
    with torch.no_grad():
        features = dinov2_model.forward_features(image.unsqueeze(0).cuda())

    patch_tokens = features["x_norm_patchtokens"]
    cls_token = features["x_norm_clstoken"].squeeze(0).cpu().numpy()

    features = patch_tokens.squeeze(0).cpu().numpy()
    num_patches = patch_tokens.shape[1]
    features_h = image.shape[1] // 14
    features_w = image.shape[2] // 14

    feature_patch_info_dict = {
        "num_patches": num_patches,
        "features": features,
        "features_h": features_h,
        "features_w": features_w,
        "orig_image": image,
        "cls_token": cls_token
    }
    # print(features.shape)
    return feature_patch_info_dict