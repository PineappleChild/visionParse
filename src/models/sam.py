from segment_anything import sam_model_registry, SamPredictor


def get_sam(
            model_name: str,
            checkpoint_path: str,
            device
            ) -> SamPredictor:
    
        sam = sam_model_registry[model_name](checkpoint=checkpoint_path)
        sam.to(device)
        return SamPredictor(sam)



