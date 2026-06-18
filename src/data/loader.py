import torch
from torchvision.datasets import ImageFolder

from pathlib import Path

from data.transforms import get_transform

def dataLoad(
            file_path: Path, 
            resize_dim: int,
            batch_size: int,
            num_workers: int,
            pin_memory: bool,
            persistent_workers: bool,
            prefetch_factor: int
            ) -> tuple:

        print(f"loading from: {file_path}")

        transform = get_transform(resize_dim=resize_dim)

        dataset = ImageFolder(root=file_path, transform=transform)

        print(f"Num samples: {len(dataset)}")

        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                persistent_workers=persistent_workers,
                                                prefetch_factor=prefetch_factor
                                                )

        return dataloader, dataset.classes