from torchvision import transforms

def get_transform(
                resize_dim: int
                ) -> transforms:
    transform = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.ToTensor(),
            # torchV.transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
    
    return transform