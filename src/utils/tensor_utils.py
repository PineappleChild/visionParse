import torch
import numpy as np

def denormalize_imagenet(
                        image_tensor: torch.Tensor
                        ) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = torch.clamp(image_tensor.cpu() * std + mean, 0, 1)
    return img.permute(1, 2, 0).numpy()