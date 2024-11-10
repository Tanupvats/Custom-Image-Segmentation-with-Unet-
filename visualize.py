# visualize.py

import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import CarPartsDataset
from model import UNetPlusPlus
import cv2
import numpy as np

def visualize_predictions(model, dataloader, device, num_classes):
    model.eval()
    images, masks = next(iter(dataloader))
    images = images.to(device)
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    masks = masks.cpu().numpy()

    # Define colors for different classes
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
        (64, 64, 0), (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0),
        (0, 0, 192), (192, 192, 0), (192, 0, 192), (0, 192, 192), (64, 128, 0),
        (128, 64, 0), (0, 64, 128), (128, 0, 64), (64, 0, 128), (0, 128, 64),
        (192, 64, 0), (64, 192, 0), (0, 64, 192), (192, 0, 64), (64, 0, 192),
        (0, 192, 64), (128, 128, 128), (64, 64, 64), (192, 192, 192), (255, 128, 0),
        (255, 0, 128), (128, 255, 0), (0, 128, 255), (128, 0, 255), (0, 255, 128),
        (255, 64, 64), (64, 255, 64), (64, 64, 255), (128, 255, 255)
    ]

    for i in range(len(images)):
        image = images[i] * 255.0  # Denormalize
        image = image.astype(np.uint8)

        # Create an overlay for predicted segmentation
        overlay = image.copy()
        for cls in range(num_classes):
            mask = (preds[i] == cls)
            overlay[mask] = colors[cls]

        # Blend the original image and the overlay
        blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(blended)
        ax.set_title('Predicted Segmentation with Labels')
        plt.axis('off')
        plt.show()
        if i == 2:
            break

def main():
    # Configuration
    num_classes = 50
    batch_size = 4
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
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = UNetPlusPlus(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_load_path))

    # Visualize predictions
    visualize_predictions(model, val_loader, device, num_classes)

if __name__ == '__main__':
    main()
