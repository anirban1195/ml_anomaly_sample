#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 22:09:38 2025

@author: dutta26
"""

import torch
import torch.nn as nn
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
from bottles_autoencoder import ConvAutoencoder

#CONFIG 
############################################################
# Path to the directory containing the test images
#IMAGE_DIR = "/scratch/bell/dutta26/mvtec/bottle/train/good/"
IMAGE_DIR ="/scratch/bell/dutta26/mvtec/bottle/test/broken_small/"
# Path to the saved model file
MODEL_PATH = "/home/dutta26/codes/div2k/ae_model/autoencoder_bottles.pth"
# Directory to save visualization results
OUTPUT_DIR = "evaluation_results"
#Image size the model was trained on
IMAGE_SIZE = 256
# Latent dimension of the model 
LATENT_DIM = 128
############################################################

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# Main Eval Script
loss_list = []
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Loading model
    model = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set the model to evaluation mode
    print("Model loaded successfully.")

    # Pixel wise MSE loss works
    criterion = nn.MSELoss()

    # Resize the full size images to the same size 
    # as the model was trained on
    eval_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    #Get list of images to evaluate
    image_files = [f for f in os.listdir(IMAGE_DIR)]
    image_files = image_files[0:10] #Sample only the first 10. 

    if not image_files:
        print(f"No PNG images found in {IMAGE_DIR}")
    else:
        print(f"Found {len(image_files)} images to evaluate.")

    # 5. Loop through images, calculate loss, and visualize
    with torch.no_grad(): # Disable gradient calculations
        for image_file in image_files:
            img_path = os.path.join(IMAGE_DIR, image_file)

            # Load and preprocess the image
            image = Image.open(img_path).convert("RGB")
            tensor_image = eval_transforms(image).unsqueeze(0).to(DEVICE) # Add batch dimension

            # Get the model's reconstruction
            reconstructed_image = model(tensor_image)

            # Calculate the L2 reconstruction loss
            loss = criterion(reconstructed_image, tensor_image)
            #print (loss)
            #loss_list.append(loss.numpy())
            #The Block below is for visualization ONLY
            #==============================================================
            print(f"Image: {image_file} | Reconstruction Loss: {loss.item():.8f}")

            # --- Visualization ---
            # Convert tensors to numpy arrays for plotting
            original = tensor_image.squeeze().cpu().permute(1, 2, 0).numpy()
            reconstructed = reconstructed_image.squeeze().cpu().permute(1, 2, 0).numpy()

            # Anomaly map (pixel-wise difference)
            anomaly_map = (original - reconstructed)**2
            anomaly_map = anomaly_map.mean(axis=2) # Average across color channels

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(original)
            ax1.set_title('Original Image')
            ax1.axis('off')

            ax2.imshow(reconstructed)
            ax2.set_title('Reconstructed Image')
            ax2.axis('off')

            ax3.imshow(anomaly_map, cmap='hot', vmax= 0.5, vmin = 0)
            ax3.set_title(f'Anomaly Map (Loss: {loss.item():.6f})')
            ax3.axis('off')

            #plt.tight_layout()
            output_path = os.path.join(OUTPUT_DIR, f"result_{image_file}")
            plt.savefig(output_path)
            plt.close()
            #====================================================================
            #End of vizualization block

    print("Evaluation complete.")
