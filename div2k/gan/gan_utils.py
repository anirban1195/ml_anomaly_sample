#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 01:01:17 2025

@author: dutta26
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from glob import glob

# Import model definitions from your training script
from wgan_gradient_penalty import Generator, Discriminator

def load_gan_models(models_dir, latent_dim, device):
    """
    Loads the pre-trained Generator and Discriminator models.
    """
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    gen_path = os.path.join(models_dir, "generator.pth")
    disc_path = os.path.join(models_dir, "discriminator.pth")

    generator.load_state_dict(torch.load(gen_path, map_location=device))
    discriminator.load_state_dict(torch.load(disc_path, map_location=device))

    #Set to eval mode
    generator.eval()
    discriminator.eval()
    return generator, discriminator

def get_gan_anomaly_details(generator, image_path, image_transforms, device, search_iterations, search_lr, reduction='mean'):
    """
    Processes a single image to find its best reconstruction and anomaly score.
    """
    #reconstruction_criterion = nn.L1Loss(reduction=reduction)
    reconstruction_criterion = nn.MSELoss(reduction=reduction) #Works better
    latent_dim = 100
    
    real_image = Image.open(image_path).convert("RGB")
    real_image = image_transforms(real_image).unsqueeze(0).to(device)

    z_optimal = torch.randn(1, latent_dim, 1, 1, device=device, requires_grad=True) #Starts with a random guess
    z_optimizer = optim.Adam([z_optimal], lr=search_lr)

    for _ in range(search_iterations):
        z_optimizer.zero_grad()
        generated_image = generator(z_optimal)
        loss = reconstruction_criterion(generated_image, real_image)
        loss.backward()
        z_optimizer.step()

    best_reconstruction = generator(z_optimal).detach()
    anomaly_score = reconstruction_criterion(best_reconstruction, real_image).item()

    real_image_viz = real_image.squeeze().cpu().numpy() * 0.5 + 0.5
    best_reconstruction_viz = best_reconstruction.squeeze().cpu().numpy() * 0.5 + 0.5
    anomaly_map = np.mean(np.abs(real_image_viz - best_reconstruction_viz), axis=0)

    return real_image, anomaly_map, float(anomaly_score)

def calculate_scores_for_dataset(generator, test_data_root, image_transforms, device, search_iterations, search_lr, reduction='mean'):
    """
    Calculates anomaly scores for all images, returning a list of dictionaries.
    """
    results = []
    defect_types = sorted(os.listdir(test_data_root))

    for defect_type in defect_types:
        defect_dir = os.path.join(test_data_root, defect_type)
        if not os.path.isdir(defect_dir): continue

        image_files = [f for f in os.listdir(defect_dir) if f.endswith(('.png'))]

        print(f"Category: {defect_type} ({len(image_files)} images) with reduction='{reduction}'")
        for image_file in image_files:
            image_path = os.path.join(defect_dir, image_file)
            _, _, score = get_gan_anomaly_details(
                generator, image_path, image_transforms, device, search_iterations, search_lr, reduction
            )
            results.append({
                'path': image_path,
                'label': defect_type,
                'is_defect': 0 if defect_type == 'good' else 1,
                'anomaly_score': score
            })
    return results

def plot_roc_curve(y_true, y_scores, output_path, title_suffix=""):
    """
    Plots the ROC curve from lists of true labels and scores.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Image-Level ROC Curve {title_suffix}')
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.savefig(output_path)
    plt.close()
    print(f"Image-level ROC curve saved to {output_path}")

def plot_score_histograms(results_data, output_path, title_suffix=""):
    """
    Plots histograms from a list of result dictionaries.
    """
    scores_by_label = {}
    for item in results_data:
        label = item['label']
        score = item['anomaly_score']
        if label not in scores_by_label:
            scores_by_label[label] = []
        scores_by_label[label].append(score)

    #Create a set of colors for each label
    labels = sorted(scores_by_label.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

    plt.figure(figsize=(12, 7))
    for label, color in zip(labels, colors):
        scores = scores_by_label[label]
        plt.hist(scores, alpha=0.7, label=label, color=color, histtype = 'step', lw= 3)

    plt.title('Distribution of Anomaly Scores')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(axis='y')

    plt.savefig(output_path)
    plt.close()
    

def plot_pixel_roc_curve(generator, test_data_root, ground_truth_root, image_transforms, device, search_iterations, search_lr, reduction, output_path, title_suffix=""):
    """
    Calculates and plots the pixel-level ROC curve and AUC.
    """
    all_scores = []
    all_labels = []

    mask_transforms = transforms.Compose([
        transforms.Resize(image_transforms.transforms[0].size),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    defective_paths = [p for p in glob(os.path.join(test_data_root, '*', '*')) if 'good' not in p]

    print("Num of defective images found is", len(defective_paths))

    for img_path in defective_paths:
        path_parts = img_path.split(os.sep)
        base_name, extension = os.path.splitext(path_parts[-1])
        defect_type = path_parts[-2]

        mask_filename = f"{base_name}_mask{extension}"
        mask_path = os.path.join(ground_truth_root, defect_type, mask_filename)

        if not os.path.exists(mask_path): 
            continue

        _, anomaly_map, _ = get_gan_anomaly_details(
            generator, img_path, image_transforms, device, search_iterations, search_lr, reduction
        )

        mask = Image.open(mask_path)
        mask_tensor = mask_transforms(mask).squeeze().cpu().numpy()
        labels = (mask_tensor > 0).astype(int)

        all_scores.append(anomaly_map.flatten())
        all_labels.append(labels.flatten())
        
    
    if not all_scores:
        print("Error!! Nothing was calculated")
        return 0.0
    
    final_scores = np.concatenate(all_scores)
    final_labels = np.concatenate(all_labels)

    fpr, tpr, _ = roc_curve(final_labels, final_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'Pixel-wise ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (per pixel)')
    plt.ylabel('True Positive Rate (per pixel)')
    plt.title(f'Pixel-Level ROC Curve {title_suffix}')
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.savefig(output_path)
    plt.close()
    print(f"Pixel-level ROC curve saved to {output_path}")





