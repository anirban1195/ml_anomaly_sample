#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 00:32:40 2025

@author: dutta26
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import random

# Configuration
########################################################
DATA_DIR = "/scratch/bell/dutta26/mvtec/bottle/train/good/"          # Directory with 'good' training images
IMAGE_SIZE = 256                 # Reduced image size for the model
BATCH_SIZE = 10                  #Not really super important in this context                
FEATURE_PERCENTAGE = 0.10        # 10% of features will be selected for the memory bank
# Path to save the final memory bank'
MEMORY_BANK_PATH = "memory_bank.pt" 

# The normalization stats from ImageNet as we're using a pre-trained model.
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
#########################################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# Custom dataset for bottles (very similar across models)
class BottleDataset(Dataset):
    """Custom Dataset for loading bottle images."""
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_files = [f for f in os.listdir(directory)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Feature Extractor Setup 
# Use hooks to grab intermediate features from the WideResNet-50.

def get_feature_extractor():
    
    # Load pre trained wide resNet-50
    model = models.wide_resnet50_2(pretrained = True)
    model = model.to(DEVICE)
    model.eval() 

    # This dictionary will store the feature maps
    features = {}

    # Creates the hook for different layers
    def create_hook(name):
        def hook(model, input, output):
            features[name] = output
        return hook

    # Register forward hooks on the layers
    # Deeper layers work well. 
    model.layer2.register_forward_hook(create_hook('layer2'))
    model.layer3.register_forward_hook(create_hook('layer3'))

    return model, features

#  Main Training Script 
if __name__ == "__main__":
    
    # Setup dataset and dataloader
    # No random augmentations needed just resizing and normalizing
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    dataset = BottleDataset(directory=DATA_DIR, transform=train_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

   

    # Initialize model and feature dictionary
    model, features_dict = get_feature_extractor()

    # --- Step 1: Feature Extraction ---
    print("Extracting features from the training set...")
    all_features = []

    with torch.no_grad():
        for images in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(DEVICE)

            # Forward pass to trigger the hooks
            model(images)

            # Get features from layer2 and layer3
            layer2_features = features_dict['layer2']
            layer3_features = features_dict['layer3']

            # Adaptive average pooling to resize
            # We resize layer2 features to match layer3's spatial dimensions
            pool = torch.nn.AdaptiveAvgPool2d(output_size=layer3_features.shape[2:])
            resized_layer2_features = pool(layer2_features)

            # Concat along the chanel dimension
            combined_features = torch.cat((resized_layer2_features, layer3_features), dim=1)

            # Reshape from [B, C, H, W] to [B*H*W, C] to get patch-level features
            # permute moves the channel dimension to the end for easier reshaping
            patch_features = combined_features.permute(0, 2, 3, 1).reshape(-1, combined_features.size(1))

            all_features.append(patch_features.cpu())

    # Concatenate All features from ALL images into a single large tensor
    full_feature_library = torch.cat(all_features, dim=0)
    print(f"Total number of feature vectors extracted: {full_feature_library.shape[0]}")

    # Step 2: 
    # Greedy Set Selection 
    print("Performing greedy coreset selection")

    # Calculate target size for the memory bank
    target_coreset_size = int(full_feature_library.shape[0] * FEATURE_PERCENTAGE)

    # Start with a random feature
    start_index = random.randint(0, full_feature_library.shape[0] - 1)
    memory_bank = full_feature_library[start_index:start_index+1]

    # Initialize minimum distances for all features to the one in the bank
    # This stores the distance of each feature to its NEAREST neighbor in the memory bank
    min_distances = torch.linalg.norm(full_feature_library - memory_bank, dim=1)

    progress_bar = tqdm(range(target_coreset_size - 1), desc="Building Memory Bank")
    for _ in progress_bar:
        # Find the feature that is furthest away from any feature currently in the memory bank
        next_feature_index = torch.argmax(min_distances).item()
        next_feature = full_feature_library[next_feature_index:next_feature_index+1]

        # Add this new feature to the memory bank
        memory_bank = torch.cat([memory_bank, next_feature], dim=0)

        # Update the minimum distances for all features
        # We only need to calculate distances to the NEWLY added feature
        new_distances = torch.linalg.norm(full_feature_library - next_feature, dim=1)
        min_distances = torch.min(min_distances, new_distances)
        progress_bar.set_postfix({"Bank Size": len(memory_bank)})

    print(f"Memory bank created with {memory_bank.shape[0]} feature vectors.")

    # Final Step:  Save the Memory Bank 
    torch.save(memory_bank, MEMORY_BANK_PATH)
    print("Training complete.")
