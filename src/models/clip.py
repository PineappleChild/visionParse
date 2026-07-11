import clip
import torch

def get_clip_model(
                    model_name: str, 
                    device
                    ):
    clip_model, clip_preprocess = clip.load(model_name, device=device)

    return clip_model, clip_preprocess

def encode_prompts(
                    prompts, 
                    clip_model, 
                    device
                    ):
        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            features = clip_model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features

def build_clip_prompts(
                        clip_model,
                        device
                        ) -> tuple:

    target_prompts = [
        "a photo of a fresh fruit",
        "a photo of a fresh vegetable",
        "a close up of produce",
        "a ripe fruit or vegetable",
        "a cross section of cut open fruit",
        "inside of fruit",
        "Leafy vegetable",
    ]

    background_prompts = [
        "a photo of a wooden table",
        "a plain white background",
        "a colored background",
        "a kitchen countertop",
        "soil and leaves outdoors",
        "a basket or bowl",
        "text of any kind",
        "leaves and foliage in the background",
        "tree branches",
        "green tree leaves",
        "tree bark and branches",
        "dried corn husk",
        "a wood cutting board",
        "stems or leaves attached to fruit",
        "stems and leaves attached to vegtables"
    ]
    
    return encode_prompts(target_prompts,clip_model, device), encode_prompts(background_prompts, clip_model, device)