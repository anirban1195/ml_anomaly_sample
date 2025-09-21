#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 01:33:13 2025

@author: dutta26
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from bottles_resnet import get_feature_extractor

# Configuration 
#######################################################################
IMAGE_DIR = "/scratch/bell/dutta26/mvtec/bottle/test/broken_large/"          # Directory with test images
MEMORY_BANK_PATH = "memory_bank.pt" # Path to the trained memory bank
OUTPUT_DIR = "resnet_results"    # Directory to save visualizations
IMAGE_SIZE = 256

# The normalization stats from ImageNet as we're using a pre-trained model.
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
#######################################################################


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
        img_name = self.image_files[idx]
        img_path = os.path.join(self.directory, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


#Main Evaluation Script

if __name__ == "__main__":
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load the Memory Bank
    print(f"Loading memory bank from {MEMORY_BANK_PATH}...")
    memory_bank = torch.load(MEMORY_BANK_PATH).to(DEVICE)


    #Dataset and DataLoader for test images
    eval_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_dataset = BottleDataset(directory=IMAGE_DIR, transform=eval_transforms)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    #Initialize the feature extractor
    model, features_dict = get_feature_extractor()

    #Process test images
    print("Starting evaluation...")
    with torch.no_grad():
        for image, image_name in tqdm(eval_dataloader, desc="Evaluating Images"):
            image_name = image_name[0] # Dataloader returns a list
            image = image.to(DEVICE)

            # Extact Features 
            model(image)
            layer2_features = features_dict['layer2']
            layer3_features = features_dict['layer3']

            pool = torch.nn.AdaptiveAvgPool2d(output_size=layer3_features.shape[2:])
            resized_layer2_features = pool(layer2_features)

            combined_features = torch.cat((resized_layer2_features, layer3_features), dim=1)
            patch_features = combined_features.permute(0, 2, 3, 1).reshape(-1, combined_features.size(1))

            # Calculate Anomaly Score for Each Patch
            # Fro each pixel in the 16x16 image we have N-D features 
            # Memory bank has M number N-D feaures 
            #So for each pixel we find the closest among M features 
            dist_matrix = torch.cdist(patch_features, memory_bank) # Shape: [N_patches, N_memory_bank]
            min_distances, _ = torch.min(dist_matrix, dim=1)

            #Create Anomaly  Map 
            #Basically calculating how anomalous each patch is in the 
            #16 x16 image by comaring it to a set of 16x16 image i.e. memory bank
            anomaly_map_shape = (1, 1, layer3_features.shape[2], layer3_features.shape[3])
            anomaly_map = min_distances.reshape(anomaly_map_shape)

           
            # Upsample to the original image size
            anomaly_map_resized = F.interpolate(
                anomaly_map, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False
            )

            # Apply Gaussian blur for smoothing
            smoother = transforms.GaussianBlur(kernel_size=5)
            anomaly_map_smoothed = smoother(anomaly_map_resized)

            # Get the image-level anomaly score (max/sum value in the map)
            #image_level_score = torch.sum(anomaly_map_smoothed).item()
            image_level_score = torch.max(anomaly_map_smoothed).item()
            print(f"Image: {image_name} | Anomaly Score: {image_level_score:.4f}")

            # Visualization
            original_image = image.squeeze().cpu().permute(1, 2, 0).numpy()
            # Un-normalize for display
            original_image = (original_image * np.array(imagenet_std)) + np.array(imagenet_mean)
            original_image = np.clip(original_image, 0, 1)

            heat_map = anomaly_map_smoothed.squeeze().cpu().numpy()

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            ax1.imshow(original_image)
            ax1.set_title('Original Image')
            ax1.axis('off')

            im = ax2.imshow(heat_map, cmap='jet', vmax = 6, vmin = 3)
            ax2.set_title('Anomaly Heat Map')
            ax2.axis('off')
            fig.colorbar(im, ax=ax2)

            ax3.imshow(original_image)
            ax3.imshow(heat_map, cmap='jet', alpha=0.5, vmax = 6, vmin = 3) # Overlay
            ax3.set_title('Overlayed Anomaly Map')
            ax3.axis('off')

            plt.suptitle(f"Image: {image_name}\nAnomaly Score: {image_level_score:.4f}", fontsize=16)
            plt.tight_layout()
            output_path = os.path.join(OUTPUT_DIR, f"result_{image_name}")
            plt.savefig(output_path)
            plt.close()

