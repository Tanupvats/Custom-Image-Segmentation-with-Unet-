
# dataset.py

import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CarPartsDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images_list = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.images_list[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = os.path.join(self.masks_dir, self.images_list[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Apply transformations
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert mask to LongTensor for CrossEntropyLoss
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
