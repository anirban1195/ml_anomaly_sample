#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 10:37:28 2025

@author: dutta26
"""

# analysis_utils.py

import torch
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from torchvision import transforms
from tqdm import tqdm

# Import your model class from your training script
from bottles_autoencoder import ConvAutoencoder

# Set seaborn style for prettier plots
sns.set_style("whitegrid")


def load_model(model_path, latent_dim, device):
    """
    Loads the pre-trained ConvAutoencoder model.

    Args:
        model_path (str): Path to the saved .pth model file.
        latent_dim (int): The latent dimension of the autoencoder.
        device (str): The device to load the model onto ('cuda' or 'cpu').

    Returns:
        torch.nn.Module: The loaded model in evaluation mode.
    """
    model = ConvAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded successfully from {model_path}")
    return model


def get_anomaly_details(model, image_path, transforms, device, aggregation_method='max'):
    """
    Processes a single image to get its reconstruction, anomaly map, and score.

    Args:
        model (torch.nn.Module): The trained autoencoder model.
        image_path (str): Path to the input image file.
        transforms (torchvision.transforms.Compose): Image transformations.
        device (str): The device the model is on.
        aggregation_method (str): Method to calculate scalar score ('max' or 'mean').

    Returns:
        tuple: A tuple containing:
            - original_img (torch.Tensor): The preprocessed original image tensor.
            - reconstructed_img (torch.Tensor): The reconstructed image tensor.
            - anomaly_map (np.ndarray): The calculated anomaly map.
            - anomaly_score (float): The scalar anomaly score.
    """
    image = Image.open(image_path).convert("RGB")
    tensor_image = transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed_image = model(tensor_image)

    # Calculate pixel-wise squared error
    squared_error = (tensor_image - reconstructed_image).squeeze().cpu().numpy()**2

    # Anomaly map is the mean error across color channels
    anomaly_map = np.mean(squared_error, axis=0)

    # Aggregate the map to get a single score
    if aggregation_method == 'max':
        anomaly_score = np.max(anomaly_map)
    elif aggregation_method == 'mean':
        anomaly_score = np.mean(anomaly_map)
    elif aggregation_method == 'sum':
        anomaly_score = np.sum(anomaly_map)
    else:
        raise ValueError("aggregation_method must be 'max' or 'mean'")

    return tensor_image, reconstructed_image, anomaly_map, float(anomaly_score)


def plot_reconstruction_and_map(original_tensor, reconstructed_tensor, anomaly_map, score, title_prefix=""):
    """
    Visualizes the original image, reconstruction, and anomaly map.
    """
    original = original_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    reconstructed = reconstructed_tensor.squeeze().cpu().permute(1, 2, 0).numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title_prefix, fontsize=16)

    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(reconstructed)
    ax2.set_title('Reconstructed Image')
    ax2.axis('off')

    im = ax3.imshow(anomaly_map, cmap='hot')
    ax3.set_title(f'Anomaly Map (Score: {score:.4f})')
    ax3.axis('off')
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def calculate_scores_for_dataset(model, test_data_root, transforms, device, aggregation_method='max'):
    """
    Calculates anomaly scores for all images in the test dataset.

    Args:
        model (torch.nn.Module): The trained autoencoder model.
        test_data_root (str): Path to the root of the test data (containing subfolders).
        transforms (torchvision.transforms.Compose): Image transformations.
        device (str): The device the model is on.
        aggregation_method (str): Method to calculate scalar score ('max' or 'mean').

    Returns:
        pd.DataFrame: A DataFrame with image paths, labels, defect status, and scores.
    """
    results = []
    defect_types = os.listdir(test_data_root)

    for defect_type in defect_types:
        defect_dir = os.path.join(test_data_root, defect_type)
        if not os.path.isdir(defect_dir):
            continue

        image_files = [f for f in os.listdir(defect_dir) if f.endswith(('.png'))]

        print(f"Processing category: {defect_type} ({len(image_files)} images)")
        for image_file in image_files:
            image_path = os.path.join(defect_dir, image_file)

            _, _, _, score = get_anomaly_details(
                model=model, image_path=image_path, transforms=transforms, device=device, aggregation_method=aggregation_method
            )

            results.append({
                'path': image_path,
                'label': defect_type,
                'is_defect': 0 if defect_type == 'good' else 1,
                'anomaly_score': score
            })

    return pd.DataFrame(results)


def plot_roc_curve(df):
    """
    Plots the ROC curve and calculates AUC from the results DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'is_defect' and 'anomaly_score' columns.
    """
    fpr, tpr, thresh = roc_curve(df['is_defect'], df['anomaly_score'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_score_histograms(df):
    """
    Plots histograms of anomaly scores, separated by defect type.

    Args:
        df (pd.DataFrame): DataFrame containing 'label' and 'anomaly_score' columns.
    """
    plt.figure(figsize=(12, 7))
    sns.histplot(data=df, x='anomaly_score', hue='label', element = 'poly', fill = False)
    plt.title('Distribution of Anomaly Scores by Defect Type')
    plt.xlabel('Anomaly Score (mean)')
    plt.ylabel('Count')
    plt.show()
    
    
    

from glob import glob

def plot_pixel_roc_curve(model, test_data_root, ground_truth_root, image_transforms, device):
    """
    Calculates and plots the pixel-level ROC curve and AUC.

    This function iterates through all defective images, compares their anomaly maps
    to the ground truth masks, and computes a pixel-wise ROC curve.
    """
    all_scores = []
    all_labels = []

    # Define a transform for the masks to ensure they match the image size
    #The convert to grayscale and then to range 0,1
    mask_transforms = transforms.Compose([
        transforms.Resize(image_transforms.transforms[0].size),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    defective_paths = [p for p in glob(os.path.join(test_data_root, '*', '*')) if 'good' not in p]

    print("Num of defective images found is", len(defective_paths))

    for img_path in defective_paths:
        path_parts = img_path.split(os.sep)
        image_filename = path_parts[-1]
        defect_type = path_parts[-2]

        base_name, extension = os.path.splitext(image_filename)
        mask_filename = f"{base_name}_mask{extension}"
        mask_path = os.path.join(ground_truth_root, defect_type, mask_filename)

        if not os.path.exists(mask_path):
            print("Mask not found for skipping", img_path)
            continue

        # Get the anomaly map for the image
        _, _, anomaly_map, _ = get_anomaly_details(model, img_path, image_transforms, device)

        mask = Image.open(mask_path)
        mask_tensor = mask_transforms(mask).squeeze().cpu().numpy()

        labels = (mask_tensor > 0).astype(int)  #Labels of whether they are defective regions or not

        all_scores.append(anomaly_map.flatten())
        all_labels.append(labels.flatten())

    if not all_scores:
        print("Error!! Nothing was calculated")
        return 0.0

    final_scores = np.concatenate(all_scores)
    final_labels = np.concatenate(all_labels)

    fpr, tpr, thresh = roc_curve(final_labels, final_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'Pixel-wise ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate(per pixel)')
    plt.ylabel('True Positive Rate (per pixel)')
    plt.title('Pixel-Level ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return roc_auc