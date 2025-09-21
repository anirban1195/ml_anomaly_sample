#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 01:22:56 2025

@author: dutta26
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from glob import glob

from bottles_resnet import get_feature_extractor

sns.set_style("whitegrid")

# Define ImageNet stats as they are used in transforms
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_resnet_essentials(memory_bank_path, device):
    """
    Loads the feature extractor model and the memory bank.

    Args:
        memory_bank_path (str): Path to the saved memory_bank.pt file.
        device (str): The device to load components onto ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The WideResNet-50 feature extractor.
            - features_dict (dict): The dictionary to be populated by hooks.
            - memory_bank (torch.Tensor): The loaded memory bank of features.
    """
    #Initialize
    model, features_dict = get_feature_extractor()

    #Load from memory bank and return both
    memory_bank = torch.load(memory_bank_path, map_location=device)
    return model, features_dict, memory_bank


def get_resnet_anomaly_details(model, features_dict, memory_bank, image_path, image_transforms, device):
    """
    Processes a single image to get its anomaly map and score using the ResNet method.

    Args:
        model, features_dict, memory_bank: Components from load_resnet_essentials.
        image_path (str): Path to the input image file.
        image_transforms (torchvision.transforms.Compose): Image transformations.
        device (str): The device the model is on.

    Returns:
        tuple: A tuple containing:
            - original_img (torch.Tensor): The preprocessed original image tensor.
            - anomaly_map (np.ndarray): The calculated, smoothed anomaly map.
            - anomaly_score (float): The scalar anomaly score (max of the map).
    """
    #Virtually identical to the eval code
    image = Image.open(image_path).convert("RGB")  
    tensor_image = image_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # --- Feature Extraction ---
        model(tensor_image)
        layer2_features = features_dict['layer2']
        layer3_features = features_dict['layer3']

        pool = torch.nn.AdaptiveAvgPool2d(output_size=layer3_features.shape[2:])
        resized_layer2_features = pool(layer2_features)
        combined_features = torch.cat((resized_layer2_features, layer3_features), dim=1)
        patch_features = combined_features.permute(0, 2, 3, 1).reshape(-1, combined_features.size(1))

        #Anomaly Calculation
        dist_matrix = torch.cdist(patch_features, memory_bank)
        min_distances, _ = torch.min(dist_matrix, dim=1)

        # Anomaly Map Generation
        map_shape = (1, 1, layer3_features.shape[2], layer3_features.shape[3])
        anomaly_map_raw = min_distances.reshape(map_shape)

        # Upsample and smooth
        anomaly_map_resized = F.interpolate(
            anomaly_map_raw, size=tensor_image.shape[2:], mode='bilinear', align_corners=False
        )
        smoother = transforms.GaussianBlur(kernel_size=5)
        anomaly_map_smoothed = smoother(anomaly_map_resized)

        anomaly_score = torch.max(anomaly_map_smoothed).item()
        final_anomaly_map = anomaly_map_smoothed.squeeze().cpu().numpy()

    return tensor_image, final_anomaly_map, float(anomaly_score)


def plot_heatmap_and_overlay(original_tensor, anomaly_map, score, title_prefix=""):
    """
    Visualizes the original image, anomaly heatmap, and an overlay.
    """
    #Un-normalize 
    original_image = original_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    original_image = (original_image * np.array(IMAGENET_STD)) + np.array(IMAGENET_MEAN)
    original_image = np.clip(original_image, 0, 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{title_prefix} (Anomaly Score: {score:.4f})", fontsize=16)

    ax1.imshow(original_image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    im = ax2.imshow(anomaly_map, cmap='jet') 
    ax2.set_title('Anomaly Heat Map')
    ax2.axis('off')
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    ax3.imshow(original_image)
    ax3.imshow(anomaly_map, cmap='jet', alpha=0.5) # Overlay 
    ax3.set_title('Overlayed Anomaly Map')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()


def calculate_scores_for_dataset(model, features_dict, memory_bank, test_data_root, image_transforms, device):
    """
    Calculates anomaly scores for all images in the test dataset using the ResNet method.
    IDENTICAL
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

            _, _, score = get_resnet_anomaly_details(
                model, features_dict, memory_bank, image_path, image_transforms, device
            )

            results.append({
                'path': image_path,
                'label': defect_type,
                'is_defect': 0 if defect_type == 'good' else 1,
                'anomaly_score': score
            })

    return pd.DataFrame(results)

# Copied plotting functions from ae analysis

def plot_roc_curve(df):
    """
    Plots the ROC curve and calculates AUC from the results DataFrame.
    IDENTICAL
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
    IDENTICAL
    """
    plt.figure(figsize=(12, 7))
    sns.histplot(data=df, x='anomaly_score', hue='label', element = 'poly', fill = False)
    plt.title('Distribution of Anomaly Scores by Defect Type')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.show()


def plot_pixel_roc_curve(model, features_dict, memory_bank, test_data_root, ground_truth_root, image_transforms, device):
    """
    Calculates and plots the pixel-level ROC curve and AUC for ResNet method.

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
        _, anomaly_map, _ = get_resnet_anomaly_details(model, features_dict, memory_bank, img_path, image_transforms, device)

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
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (per pixel)')
    plt.ylabel('True Positive Rate (per pixel)')
    plt.title('Pixel-Level Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return roc_auc


