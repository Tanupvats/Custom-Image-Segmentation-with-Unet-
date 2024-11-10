
# train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import CarPartsDataset
from model import UNetPlusPlus

def train_model():
    # Configuration
    num_classes = 50
    num_epochs = 50
    batch_size = 8
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    train_images_dir = 'path/to/train/images'
    train_masks_dir = 'path/to/train/masks'
    val_images_dir = 'path/to/val/images'
    val_masks_dir = 'path/to/val/masks'
    model_save_path = 'unetpp_best_model.pth'

    # Transformations
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Datasets and DataLoaders
    train_dataset = CarPartsDataset(train_images_dir, train_masks_dir, transform=train_transform)
    val_dataset = CarPartsDataset(val_images_dir, val_masks_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model, Loss, Optimizer
    model = UNetPlusPlus(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        print(f"Epoch {epoch+1}/{num_epochs}")

        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)

        val_loss = val_loss / len(val_loader.dataset)

        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save the model checkpoint if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print("Model saved!")

if __name__ == '__main__':
    train_model()
