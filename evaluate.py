
# evaluate.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import CarPartsDataset
from model import UNetPlusPlus

def compute_iou(outputs, masks, num_classes):
    outputs = torch.argmax(outputs, dim=1)
    ious = []
    for cls in range(num_classes):
        intersection = ((outputs == cls) & (masks == cls)).sum().item()
        union = ((outputs == cls) | (masks == cls)).sum().item()
        if union == 0:
            ious.append(np.nan)  # Ignore if no ground truth
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

def evaluate_model():
    # Configuration
    num_classes = 50
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    val_images_dir = 'path/to/val/images'
    val_masks_dir = 'path/to/val/masks'
    model_load_path = 'unetpp_best_model.pth'

    # Transformations
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Dataset and DataLoader
    val_dataset = CarPartsDataset(val_images_dir, val_masks_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model and Loss
    model = UNetPlusPlus(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_load_path))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # Evaluation Loop
    val_loss = 0.0
    val_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item() * images.size(0)
            val_iou += compute_iou(outputs, masks, num_classes) * images.size(0)

    val_loss = val_loss / len(val_loader.dataset)
    val_iou = val_iou / len(val_loader.dataset)

    print(f"Validation Loss: {val_loss:.4f}, Mean IoU: {val_iou:.4f}")

if __name__ == '__main__':
    evaluate_model()
