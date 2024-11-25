import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FabricDataset(Dataset):
    def __init__(self, noisy_images_dir, clean_images_dir, transform=None):
        self.noisy_images_dir = noisy_images_dir
        self.clean_images_dir = clean_images_dir
        self.image_files = [f for f in os.listdir(noisy_images_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        noisy_image = Image.open(os.path.join(self.noisy_images_dir, img_name)).convert('RGB')
        clean_image = Image.open(os.path.join(self.clean_images_dir, img_name)).convert('RGB')

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

def get_data_loaders(train_dir, test_dir, batch_size=4):
    train_dataset = FabricDataset(
        noisy_images_dir=os.path.join(train_dir, 'noisy'),
        clean_images_dir=os.path.join(train_dir, 'clean'),
        transform=transform
    )
    test_dataset = FabricDataset(
        noisy_images_dir=os.path.join(test_dir, 'noisy'),
        clean_images_dir=os.path.join(test_dir, 'clean'),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader