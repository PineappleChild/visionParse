import torch

from pathlib import Path

from data.loader import dataLoad

from pipeline.pipeline import Pipeline

# clustering is the biggest bottneck in getting good regions and predictions for what is not part of the target prompts
#smoother and less chunky would be good, also sometimes it bleeds in and causes it to change the scores for the background and 
#mixed in target area resulting in a poor score

path_to_train = Path(r"E:\featureLearningJourney\LatentSpaceFruitsAndVeggies\train")

sam_checkpoint_path = Path(r"E:\featureLearningJourney\LatentSpaceFruitsAndVeggies\auto_segment_dataset\sam_vit_b_01ec64.pth")

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

    pipeline = Pipeline(sam_checkpoint_path=sam_checkpoint_path, data_loader=dataloader_train, device=device)
    pipeline.run()
