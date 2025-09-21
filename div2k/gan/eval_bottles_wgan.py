#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 11:00:15 2025

@author: dutta26
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from wgan_gradient_penalty import Generator, Discriminator

#Configuration
#######################################################################
IMAGE_DIR = "/scratch/bell/dutta26/mvtec/bottle/test/broken_small/"   # Directory with test images
MODELS_DIR = "gan_models"       # Directory where trained models are saved
OUTPUT_DIR = "gan_results"      # Directory to save visualizations
IMAGE_SIZE = 256
LATENT_DIM = 100

# Hyperparameters for the latent space search
SEARCH_ITERATIONS = 500         # Number of steps to find the best z
SEARCH_LR = 0.01                # Learning rate for z optimization
LAMBDA_RECONSTRUCTION = 1     # Weight for reconstruction score in final score
#LAMBDA_DISCRIMINATOR = 0.1
#######################################################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

#Reused from training bottles dataset
class BottleDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_files = [f for f in os.listdir(directory)]
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.directory, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, img_name






if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #Load trained models
    generator = Generator(LATENT_DIM).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    generator.load_state_dict(torch.load(os.path.join(MODELS_DIR, "generator.pth"), map_location=DEVICE))
    discriminator.load_state_dict(torch.load(os.path.join(MODELS_DIR, "discriminator.pth"), map_location=DEVICE))
    generator.eval()
    discriminator.eval()
    print("Models loaded successfully.")

    #Dataet and transformation setup
    eval_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    eval_dataset = BottleDataset(directory=IMAGE_DIR, transform=eval_transforms)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # 3. Define the loss function for reconstruction
    #reconstruction_criterion = nn.L1Loss(reduction='sum')
    reconstruction_criterion = nn.L1Loss(reduction='mean')


    #Process test images
    for real_image, image_name in eval_dataloader:
        image_name = image_name[0]
        real_image = real_image.to(DEVICE)

        # Search latent sapce
        # Initialize a random z and optimize
        z_optimal = torch.randn(1, LATENT_DIM, 1, 1, device=DEVICE, requires_grad=True)
        z_optimizer = optim.Adam([z_optimal], lr=SEARCH_LR)

        for _ in range(SEARCH_ITERATIONS):
            z_optimizer.zero_grad()
            generated_image = generator(z_optimal)
            loss = reconstruction_criterion(generated_image, real_image)
            loss.backward()
            z_optimizer.step()

        
        best_reconstruction = generator(z_optimal).detach()

        # Reconstruction Score (L1 distance)
        score_r = torch.mean(torch.abs(real_image - best_reconstruction))

        #TODO: Try a feature distance based critic score.
        # Discrimination Score. Get a probability like score
        #score_d = torch.sigmoid(discriminator(real_image)) 
        #raw_discriminator_output = discriminator(real_image)
        #print(f"Raw Discriminator Score: {raw_discriminator_output.item():.4f}") 
        #score_d = torch.sigmoid(raw_discriminator_output) # Get a probability


        # Combined Anomaly Score
        anomaly_score = (LAMBDA_RECONSTRUCTION * score_r) #+ (LAMBDA_DISCRIMINATOR * (1 - score_d))
        #print (score_r, score_d, anomaly_score)
        print(
            f"Image: {image_name} | Anomaly Score: {anomaly_score.item():.4f} "
            f"Recon: {score_r.item():.4f}"
        )


        # Mkae images for display in range [0,1]
        real_image_viz = real_image.squeeze().cpu().numpy() *0.5 + 0.5
        best_reconstruction_viz = best_reconstruction.squeeze().cpu().numpy() *0.5 + 0.5

        # Anomaly map is the absolute difference
        anomaly_map = np.abs(real_image_viz - best_reconstruction_viz)
        anomaly_map = np.mean(anomaly_map, axis=0) # Convert to grayscale

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        ax1.imshow(np.transpose(real_image_viz, (1, 2, 0)))
        ax1.set_title('Original Image')
        ax1.axis('off')

        ax2.imshow(np.transpose(best_reconstruction_viz, (1, 2, 0)))
        ax2.set_title('Reconstructed Image')
        ax2.axis('off')

        im = ax3.imshow(anomaly_map, cmap='jet')
        ax3.set_title('Anomaly Heat Map')
        ax3.axis('off')
        fig.colorbar(im, ax=ax3)

        plt.suptitle(f"Image: {image_name}\nAnomaly Score: {anomaly_score.item():.4f}", fontsize=16)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f"result_{image_name}")
        plt.savefig(output_path)
        plt.close()

    print("Evaluation complete.")

