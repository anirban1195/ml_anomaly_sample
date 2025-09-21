#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 08:47:40 2025

@author: dutta26
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Model Configuration

# Path to the directory containing the 'good' training images
DATA_DIR = "/scratch/bell/dutta26/mvtec/bottle/train/good/"
# Image size to resize to for training
IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 100
# Dimension of the latent space bottleneck 
LATENT_DIM = 128
LEARNING_RATE = 1e-3
# Percentage of data to use for validation
VALIDATION_SPLIT = 0.2
# Save the model to this path 
MODEL_PATH = "/home/dutta26/codes/div2k/ae_model/autoencoder_bottles.pth"
# Save loss curve to this path 
LOSS_PATH = "/home/dutta26/codes/div2k/ae_model/loss_curve.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# We define the transformations and the custom Dataset class.
#Random flip and roatation. Also add random translation ?
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(
        degrees=360,
        translate=(0.02, 0.02),
        fill= 255 # White background fill
    ), #Double check here. FILL 1?
    transforms.ToTensor(), # Converts PIL image to tensor and scales to [0, 1]
])


#For efficiently loading the images 
class BottleDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_files = [f for f in os.listdir(directory)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_files[idx])
        # Open as RGB (to ensure 3 channels)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


#Model Definition
# Simple Convolutional Autoencoder.

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvAutoencoder, self).__init__()

        # Simple Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # -> 32x128x128
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 64x64x64
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> 128x32x32
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> 256x16x16
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, latent_dim) # -> latent_dim
        )

        # Simple Decoder
        self.decoder_projection = nn.Linear(latent_dim, 256 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 16, 16)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 128x32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> 64x64x64
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> 32x128x128
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), # -> 3x256x256
            nn.Sigmoid() # To ensure scale output pixels to [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder_projection(x)
        x = self.decoder(x)
        return x


# The main script to set up and run the training process.

if __name__ == "__main__":
    # Setup dataset and dataloaders
    full_dataset = BottleDataset(directory=DATA_DIR, transform=train_transforms)

    # Split test train
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Total images:", len(full_dataset))
    print("Training set size:", len(train_dataset))
    print("Validation set size:", len(val_dataset))

    # Initialize model, optimizer, and loss function
    model = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss() # L2 Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize LR scheduler to reduce learning rate on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    # Training Loop
    train_losses = []
    val_losses = []
    min_loss = np.inf
    for epoch in range(EPOCHS):
        # Train
        model.train()
        running_train_loss = 0.0
        for images in train_loader:
            images = images.to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images) 

            # Backward pass and optimize
            loss.backward()
            optimizer.step()


            #running_train_loss += loss.item() * BATCH_SIZE
            running_train_loss += loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, images)
                running_val_loss += loss.item() * images.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

        #Save only the pest performing model
        if (epoch_val_loss < min_loss):
            min_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("Best model found and saved to defined path")
            
        # Step the scheduler based on validation loss
        scheduler.step(epoch_val_loss)

    print("Finished Training.")

    
    # Plot and save the loss 
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Training & Validation Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_PATH)
    

    