#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 09:17:19 2025

@author: dutta26
"""
import torch
from torchvision import transforms
from torchvision.utils import make_grid
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Configuration ---
DATA_DIR = "/scratch/bell/dutta26/mvtec/bottle/train/good/"      # Directory with 'good' training images
OUTPUT_FILE = "augmentation_samples.png" # Output file name
NUM_SAMPLES = 16             # Number of augmented samples to generate
GRID_SIZE = 4                # We'll create a 4x4 grid of samples

# --- 2. Reused Transforms from GAN training script ---
# This should be the EXACT same transform pipeline to ensure a valid check.
IMAGE_SIZE = 128
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    #transforms.RandomRotation(degrees=360, fill=1), #Double check here. FILL 1?
    transforms.RandomAffine(
        degrees=360,
        translate=(0.08, 0.08),
        fill=255 # White background fill
    ),
    transforms.ToTensor(),
    # Tanh activation in Generator outputs images in range [-1, 1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def unnormalize(tensor):
    """Reverses the normalization for display purposes."""
    tensor = tensor.clone() # Avoid modifying the original tensor
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    # The operation is: tensor * std + mean
    # We need to reshape mean and std to broadcast correctly
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor

# --- 3. Main Visualization Script ---
if __name__ == "__main__":
    # Find the first PNG image in the directory to use as our sample
    try:
        sample_image_name = next(f for f in os.listdir(DATA_DIR) if f.endswith('.png'))
        sample_image_path = os.path.join(DATA_DIR, sample_image_name)
        print(f"Using sample image: {sample_image_name}")
    except StopIteration:
        print(f"Error: No PNG images found in '{DATA_DIR}'. Please check the path.")
        exit()

    # Load the original image
    original_image = Image.open(sample_image_path).convert("RGB")

    # Generate a list of augmented samples
    augmented_samples = []
    for _ in range(NUM_SAMPLES):
        augmented_image = train_transforms(original_image)
        augmented_samples.append(augmented_image)

    # Un-normalize the images for display
    unnormalized_samples = [unnormalize(sample) for sample in augmented_samples]

    # Create a grid of images
    image_grid = make_grid(unnormalized_samples, nrow=GRID_SIZE)

    # Convert tensor to numpy for plotting
    np_image = image_grid.numpy()

    # Plot and save the grid
    plt.figure(figsize=(10, 10))
    # Matplotlib expects channels-last format (H, W, C)
    plt.imshow(np.transpose(np_image, (1, 2, 0)))
    plt.title(f"{NUM_SAMPLES} Randomly Augmented Samples")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)

    print(f"Saved {NUM_SAMPLES} augmentation samples to '{OUTPUT_FILE}'.")
    plt.show()

